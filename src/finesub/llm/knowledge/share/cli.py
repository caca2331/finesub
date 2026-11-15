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

from finesub import jsonl
from finesub.reporting import current_reporter

from ..base import knowledge_root_path
from ..node.repo import AmbiguousName, KnowledgeRepo
from . import client, conflicts as conflict_ledger
from .exchange import (
    CORPUS_ID_RE,
    build_push_bundle,
    bundle_content_digest,
    strip_local_maps,
)
from .mailbox import default_remote, trusted_signing_key
from .relay import RELAY_BATCH_LIMIT
from .sync import anchor_keys, apply_snapshot

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


def _remote_url(url: str) -> str:
    return url.rstrip("/")


def _url_key(url: str) -> str:
    return f"share:url:{url}"


def _token_key(remote: str) -> str:
    return f"share:{remote}:token"


def _corpus_is_known(store: Any, corpus: str) -> bool:
    """Does this store already hold something that belongs to that corpus —
    a contributor token, an anti-rollback anchor, or merge state?"""

    rev_key, _ = anchor_keys(corpus)
    if store.meta(_token_key(corpus)) or store.meta(rev_key):
        return True
    row = store.conn.execute(
        "SELECT 1 AS hit FROM sync_state WHERE remote=? LIMIT 1", (corpus,)
    ).fetchone()
    return row is not None


def _remote_id(store: Any, url: str, *, accept_move: bool = False) -> str:
    """Address → corpus identity, asked once per URL and then remembered.

    Every piece of local state is keyed on the CORPUS, never on the address:
    the token, the anti-rollback anchor, ``sync_state``'s three-way merge base,
    and the two JSONL ledgers. So the library can move — workers.dev to a VPS,
    main site to a mirror — without all of that silently falling on the floor
    (a lost base is the expensive one: it does not error, it just turns the
    next pull into a screenful of phantom conflicts).

    **A corpus id is public** — it rides in every snapshot — so knowing one
    proves nothing about who is answering. Identity therefore says *which*
    library this is; it never says *this address may act for it*. That second
    question is what the address→corpus mapping in ``meta`` answers, and it is
    only ever written for an address the user pointed at deliberately:

    The gate is therefore one question — *am I about to hand this address a
    corpus whose state is already here?* — and it does not care whether the
    address is new or merely changed its answer:

    - a corpus this store holds nothing for is a new relationship: trust on
      first use, the same as the very first pull ever. A changed answer is said
      out loud, because it is worth knowing, but it costs nothing;
    - a corpus this store *does* hold a token, anchor or merge state for is
      refused unless the user passes ``--accept-move``. Otherwise a mirror that
      copied a public corpus id would be handed the contributor token minted
      for the real server (reviewer, 2026-09-09) and could merge into the
      entries the real server's snapshots created.
    """

    url = _remote_url(url)
    cached = store.meta(_url_key(url))
    try:
        reported = str(client.fetch_identity(url).get("corpus_id") or "")
    except client.RemoteError:
        if cached:
            return cached  # offline with a known address: local state is enough
        raise
    if not CORPUS_ID_RE.fullmatch(reported):
        raise SystemExit(f"error: {url} reported an unusable corpus id {reported[:40]!r}")
    if cached != reported and not accept_move:
        if _corpus_is_known(store, reported):
            seen = "an address this store has never used" if not cached else (
                f"known here as corpus {cached[:12]}…, and now"
            )
            raise SystemExit(
                f"error: {url} is {seen} claiming to be corpus {reported[:12]}… — one you already"
                " hold a token, an anchor or merge state for. A corpus id is public, so claiming"
                " one does not entitle an address to that library's credentials. If this really is"
                " the same library at a new address, re-run with --accept-move; otherwise it is an"
                " impostor and your token stays here"
            )
        if cached:
            # Not a credential risk — there is nothing here for that corpus —
            # but the user did not ask for this: the address they trusted
            # changed underneath them, and going along with it silently would
            # merge a stranger's library into their store on the next pull.
            raise SystemExit(
                f"error: {url} now answers as corpus {reported[:12]}…, not the {cached[:12]}…"
                " this store has been using — that is a different library, not a moved one."
                " Re-run with --accept-move if the address really did change hands"
            )
    if cached and cached != reported:
        current_reporter().warning(
            "share-address-changed-corpus",
            f"{url} now answers as corpus {reported[:12]}…, not the {cached[:12]}… this store"
            " saw before, so it is treated as a different library",
            action="check the address is the one you meant before pulling from it again",
        )
    if cached != reported:
        store.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (_url_key(url), reported)
        )
    return reported


def _known_remote_id(store: Any, value: str) -> str:
    """Offline resolution for the local-only commands: a URL this store has
    already talked to, or a corpus id typed directly."""

    cached = store.meta(_url_key(_remote_url(value)))
    if cached:
        return cached
    if CORPUS_ID_RE.fullmatch(value):
        return value
    raise SystemExit(
        f"error: this store has never talked to {value} — pass a corpus id, or run"
        " `share register`/`share pull` against that address first"
    )


def _cmd_register(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    url = _remote_url(args.remote or default_remote())
    remote = _remote_id(repo.store, url, accept_move=args.accept_move)
    token = client.register(url)
    repo.store.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (_token_key(remote), token),
    )
    return [f"registered with corpus {remote[:12]}…; token stored in the knowledge store"]


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
    jsonl.append(root / PUSH_LOG_FILENAME, record)


def _cmd_push(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    store = repo.store
    url = _remote_url(args.remote or default_remote())
    remote = _remote_id(store, url, accept_move=args.accept_move)
    token = store.meta(_token_key(remote))
    if not token:
        raise SystemExit(f"error: not registered with {url} — run `share register` first")
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
    #
    # Only a retry by the SAME contributor: the server binds a key to whoever
    # first pushed it. A token that was reclaimed (registered long ago, only
    # expired or rejected pushes) and replaced by `share register` is a new
    # contributor, and reusing the old key would be refused with 409 for good
    # instead of reviving the expired push under the new identity.
    from .server import contributor_id

    pusher = contributor_id(token)
    prior = None
    for record in jsonl.iter_records(repo.root / PUSH_LOG_FILENAME):
        if (
            record.get("remote") == remote
            and record.get("bundle_digest") == wire_digest
            and record.get("pusher") == pusher
        ):
            prior = record
    if prior is not None:
        bundle["idempotency_key"] = wire["idempotency_key"] = prior["idempotency_key"]
    else:
        _append_push_record(
            repo.root,
            {
                "remote": remote,
                "remote_url": url,  # display only; identity is the corpus id
                "pusher": pusher,
                "status": "intent",
                "idempotency_key": bundle["idempotency_key"],
                "bundle_digest": wire_digest,
                "handle_map": bundle["handle_map"],
                "item_handle_map": bundle["item_handle_map"],
            },
        )
    reply = client.push_bundle(url, wire, token=token)
    _append_push_record(
        repo.root,
        {
            "remote": remote,
            "remote_url": url,
            "pusher": pusher,
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
        f" check with: share status --remote {url} --queue-id {reply['queue_id']}"
    ]


def _push_record(root: Path, remote: str, queue_id: int) -> dict[str, Any] | None:
    found = None
    for record in jsonl.iter_records(root / PUSH_LOG_FILENAME):
        if record.get("remote") == remote and record.get("queue_id") == queue_id:
            found = record  # last record wins
    return found


def _cmd_status(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    url = _remote_url(args.remote or default_remote())
    remote = _remote_id(repo.store, url, accept_move=args.accept_move)
    token = repo.store.meta(_token_key(remote))
    if not token:
        raise SystemExit(f"error: not registered with {url} — run `share register` first")
    reply = client.push_status(url, args.queue_id, token=token)
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
    url = _remote_url(args.remote or default_remote())
    remote = _remote_id(repo.store, url, accept_move=args.accept_move)
    # The anti-rollback anchor doubles as the conditional-request tag: if the
    # server's chain head still matches it, there is provably nothing new, and
    # neither side pays for the corpus.
    _, hash_key = anchor_keys(remote)
    snapshot = client.fetch_snapshot(url, known_chain_hash=repo.store.meta(hash_key) or "")
    if snapshot is client.UNCHANGED:
        # Returns before `refresh_rendered()`, deliberately: nothing changed, so
        # there is nothing to project. The one behaviour this drops is that an
        # empty pull used to restore `rendered/` files deleted by hand — that
        # was a side effect of a no-op, not a promise, and `knowledge repair`
        # is where rebuilding the projection belongs.
        return [f"already at the server's current revision ({url})"]
    if str(snapshot.get("corpus_id") or "") != remote:
        # /identity is unauthenticated; the snapshot's id is the one the chain
        # commits to. Disagreement means the two answers came from different
        # corpora — refuse rather than key this merge on the wrong one.
        raise SystemExit(f"error: {url} served a snapshot from a different corpus than it announced")
    report = apply_snapshot(
        repo.store, snapshot, remote=remote, label=url,
        trusted_key=trusted_signing_key(url), accept_new_key=args.accept_new_key,
    )
    repo.refresh_rendered()
    lines = [] if snapshot.get("signature") else [
        f"warning: {url} does not sign its snapshots — this pull proves only that the server"
        " sent it, not that its maintainer published it"
    ]
    lines += [
        f"pulled server rev {report.server_rev} → local rev {report.rev}:"
        f" +{report.created_nodes} nodes, ~{report.updated_nodes} updated,"
        f" -{report.retired_nodes} retired; +{report.created_items} items,"
        f" +{report.created_memberships} memberships, {report.moved_memberships} moved,"
        f" +{report.created_links} links"
    ]
    for conflict in report.conflicts:
        lines.append(f"conflict (local kept): {conflict.describe()}")
    if report.diverged_moves:
        lines.append(
            f"local kept: {report.diverged_moves} line(s) currently moved both here and in the library,"
            " to different entries"
        )
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
            f" list them with: share conflicts --remote {url}"
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
    remote = _known_remote_id(repo.store, args.remote) if args.remote else ""
    rows = conflict_ledger.open_conflicts(repo.root, remote=remote)
    if not rows:
        return ["no open conflicts" + (f" for {args.remote}" if remote else "")]
    from .conflict_repair import human_only

    # Pending hand edits in rendered/ go in BEFORE the session reads the
    # entry (owner 2026-10-02): an unharvested file is a whole-entry diff that
    # the next harvest would apply on top of the session's answer, putting the
    # old H1 or intro straight back. Same entry point as the correction run's
    # start-of-run harvest (worktree gate, write lock, per-file isolation).
    lines = _harvest_before_repair(repo, execute=args.execute) if args.repair else []
    lines.append(f"{len(rows)} open conflict(s):")
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

    from ...client import OutputCutOff
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
        try:
            result = run_conflict_repair_session(
                repo, subject_id, group, client=client_, apply=args.apply
            )
        except OutputCutOff as exc:
            # Subjects before this one are already applied: say so, then stop.
            # Stop, not skip-and-continue: one subject's reply only reaches the
            # model's own ceiling (65k on Gemini) when it degenerates, which is
            # rare enough that a skip would cost an exit-code scheme for nothing.
            raise SystemExit("\n".join([*lines, f"error: {exc}"])) from exc
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


def _harvest_before_repair(repo: KnowledgeRepo, *, execute: bool) -> _Lines:
    """Absorb pending `rendered/` edits before a conflict-repair session.

    Dry-run (no ``--execute``) only says which files are waiting; it writes
    nothing. A file that fails to parse stays as it is and is reported by the
    harvest itself -- the session still runs."""

    dirty = repo.rendered_dirty_files()
    if not dirty:
        return []
    names = "、".join(rel for rel, _path, _subject in dirty)
    if not execute:
        return [f"rendered/ 里有 {len(dirty)} 个文件的手工改动还没收进库（{names}）；--execute 时会先收进来再开会话"]
    from ..node.edit import harvest_rendered_edits_at_run_start

    try:
        results = harvest_rendered_edits_at_run_start(repo.root)
    except Exception as exc:  # the harvest must never sink the repair
        current_reporter().warning("knowledge-rendered-harvest-failed", str(exc))
        return [f"rendered/ 的手工改动收割失败，照原样保留：{exc}"]
    return [
        f"rendered/{row['file']}: {row['status']}"
        + (f" (rev {row['rev']})" if row.get("rev") is not None else "")
        for row in results
    ]


def _cmd_review(args: argparse.Namespace) -> _Lines:
    """Maintainer review (plan §6.3). Dry-run by default and **read-only**:
    it peeks the queue without leasing (looking must not lock, round 7) and
    renders the threshold pre-pass + review prompt. --execute leases ONE item
    at a time, runs the LLM session, and releases the lease if anything
    fails; --post additionally posts the verdict — the two are separate acts
    on purpose, the verdict stays the maintainer's. --override-thresholds
    lets the maintainer approve past the §6.3 gate, with the reason recorded
    server-side."""

    from .review import UnusableReviewReply, render_review_prompt, run_review_session, threshold_report

    if (args.post or args.override_thresholds) and args.queue_id is None:
        # A verdict is one decision about one item. Posting a whole peek in
        # one go let a session's conclusions -- which contributed text can
        # steer -- land in the public library unread, and an override written
        # for one item applied to every item in the batch.
        raise SystemExit("error: --post and --override-thresholds need --queue-id (one item at a time)")
    # Maintainer side: the address is all it needs — the review CLI keeps no
    # per-corpus local state, so it never asks who is answering.
    remote = _remote_url(args.remote)
    peeked = client.peek_queue(
        remote, maintainer_token=_secret_arg(args, "maintainer_token"), queue_id=args.queue_id
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
        lines.extend(_bundle_identity_lines(peek_item.get("bundle") or {}))
        for row in threshold_report(peek_item.get("bundle") or {}):
            state = "OK" if row["satisfied"] else ("需外部印证" if row["needs_external"] else "人工")
            lines.append(f"  claim {row['node']} {row['field_path']} [{row['slot']}]: {state}")
        if not args.execute:
            prompt = render_review_prompt(peek_item)
            lines.append(f"  dry-run: review prompt rendered ({len(prompt)} chars); --execute 跑审核会话")
            continue
        leased = client.lease_queue(
            remote, maintainer_token=_secret_arg(args, "maintainer_token"),
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
            try:
                review = run_review_session(item, client=llm_client)
            except UnusableReviewReply as exc:
                # One unusable reply (no <review_verdict> block, broken JSON,
                # an unknown verdict) is this item's failure, not the batch's:
                # say so and go on; the lease is released below.
                lines.append(f"  session reply unusable ({exc}); lease released")
                continue
            lines.append(
                f"  session verdict: {review['verdict']}"
                + (f" merge={review['merge']}" if review["merge"] else "")
                + f" — {review['reason']}"
            )
            for row in review["external_evidence"]:
                lines.append(f"  evidence {row['node']} {row['field_path']}: {row['url']}")
            if review["unknown_claims"]:
                lines.append(
                    "  evidence dropped, unknown claim handle: " + ", ".join(review["unknown_claims"])
                )
            if args.post and not _confirmed_post(args, queue_id, review, lines):
                lines.append("  not posted (not confirmed); lease released")
            elif args.post:
                reply = client.post_verdict(
                    remote,
                    maintainer_token=_secret_arg(args, "maintainer_token"),
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
                        remote, maintainer_token=_secret_arg(args, "maintainer_token"),
                        queue_id=queue_id, lease_token=item["lease_token"],
                    )
                except Exception:
                    pass  # a lost release just lets the lease time out
    return lines


#: Payload fields that say WHAT a node is. The threshold rows name claims by
#: handle and field path only, so without these a strange entry name -- the
#: very thing that once became a file path -- never reached the reviewer's eyes.
_IDENTITY_FIELDS = ("surface", "zh", "category", "label", "field", "value", "target")


def _bundle_identity_lines(bundle: dict) -> _Lines:
    lines: _Lines = []
    for node in bundle.get("nodes") or []:
        payload = node.get("payload") or {}
        shown = ", ".join(
            f"{name}={json.dumps(str(payload[name]), ensure_ascii=False)}"
            for name in _IDENTITY_FIELDS
            if payload.get(name)
        )
        anchor = f" (updates {node['canonical_id']})" if node.get("canonical_id") else " (new)"
        lines.append(f"  node {node.get('handle')} [{node.get('kind')}]{anchor}: {shown}")
    return lines


def _confirmed_post(args: argparse.Namespace, queue_id: int, review: dict, lines: _Lines) -> bool:
    """The maintainer's own yes before a verdict goes public. The session's
    evidence URLs count toward the §6.3 gate once posted, so confirming here
    is the maintainer vouching for them, not the model."""

    if args.yes:
        return True
    if not sys.stdin.isatty():
        lines.append("  refusing to post without a terminal to confirm on (--yes to post anyway)")
        return False
    # What was gathered so far is the context of the question, so it goes out
    # with it rather than after the answer.
    shown = "\n".join(lines)
    lines.clear()
    answer = input(
        f"{shown}\npost verdict {review['verdict']!r} for queue item {queue_id}"
        f" (I checked the evidence URLs above)? [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}


def _secret_arg(args: argparse.Namespace, name: str) -> str:
    """A secret from `--<name>` or from `--<name>-file`.

    A long-lived secret on a command line ends up in shell history and, for a
    moment, in the process table — the file form exists so the maintainer's
    routine does not have to.
    """

    path = getattr(args, f"{name}_file", None)
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    return str(getattr(args, name) or "")


def _cmd_relay(args: argparse.Namespace) -> _Lines:
    """Maintainer: shuttle between the public mailbox and the local queue.

    The mailbox (``deploy/share-worker/``) judges nothing, so this is where the
    contributed bundles meet the admission boundary — through the very same
    ``ShareService.push`` the self-hosted server uses. The maintainer token this
    asks for is the **mailbox's**; the local service's own token never leaves
    this process, because the relay calls it in-process rather than over HTTP.
    """

    import secrets

    from .relay import run_relay
    from .server import ShareService
    from .signing import load_private_key

    if args.publish and not args.signing_key_file:
        # The signature is what makes a snapshot the maintainer's rather than
        # whatever the mailbox's token holder published; pulls from a library
        # that signed before refuse an unsigned one.
        raise SystemExit(
            "error: publishing needs --signing-key-file (create one with `share keygen`),"
            " or pass --no-publish to only move queue items"
        )
    signing_key = load_private_key(args.signing_key_file) if args.signing_key_file else None
    service = ShareService(
        args.server_root, maintainer_token=secrets.token_urlsafe(16), signing_key=signing_key
    )
    try:
        report = run_relay(
            service,
            _remote_url(args.remote or default_remote()),
            maintainer_token=_secret_arg(args, "mailbox_token"),
            limit=args.limit,
            publish=args.publish,
            purge_days=args.purge_days,
        )
    finally:
        service.close()
    return report.lines()


def _cmd_keygen(args: argparse.Namespace) -> _Lines:
    from .signing import generate_key_file

    public_key = generate_key_file(args.out)
    return [
        f"private key written to {args.out} — keep it on this machine and back it up;"
        " losing it means clients must be told a new key",
        f"public key: {public_key}",
        "give clients the public key: `[share] signing_key` in config.toml, or"
        " `BUILTIN_SIGNING_KEY` in share/mailbox.py for the built-in mailbox",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m finesub.llm.knowledge.share",
        description="Share client: mark/push/pull against a share server (plan §6).",
    )
    parser.add_argument("--root", default=None, help="knowledge root (default: resolved runtime root)")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="obtain an anonymous contributor token")
    register.add_argument(
        "--remote", default=None, help="mailbox address (default: the packaged one)"
    )
    register.add_argument("--accept-move", action="store_true", help="this address is my known library at a new address — authorize it to receive that corpus's token and merge state")
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
    push.add_argument(
        "--remote", default=None, help="mailbox address (default: the packaged one)"
    )
    push.add_argument("--accept-move", action="store_true", help="this address is my known library at a new address — authorize it to receive that corpus's token and merge state")
    push.set_defaults(func=_cmd_push)

    status = sub.add_parser("status", help="check a queued push; backfill canonical ids on approval")
    status.add_argument(
        "--remote", default=None, help="mailbox address (default: the packaged one)"
    )
    status.add_argument("--accept-move", action="store_true", help="this address is my known library at a new address — authorize it to receive that corpus's token and merge state")
    status.add_argument("--queue-id", type=int, required=True)
    status.set_defaults(func=_cmd_status)

    pull = sub.add_parser("pull", help="fetch, verify and merge the server snapshot")
    pull.add_argument(
        "--remote", default=None, help="mailbox address (default: the packaged one)"
    )
    pull.add_argument("--accept-move", action="store_true", help="this address is my known library at a new address — authorize it to receive that corpus's token and merge state")
    pull.add_argument(
        "--accept-new-key", action="store_true",
        help="the library's maintainer changed signing keys: trust the key this snapshot is signed by",
    )
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
    review_token = review.add_mutually_exclusive_group(required=True)
    review_token.add_argument("--maintainer-token")
    review_token.add_argument(
        "--maintainer-token-file",
        help="read it from a file instead, so the secret stays out of shell history",
    )
    review.add_argument("--queue-id", type=int, default=None, help="review one specific item")
    review.add_argument("--lease-seconds", type=int, default=900)
    review.add_argument(
        "--execute", action="store_true",
        help="run the review LLM session (spends quota; default only renders the prompt)",
    )
    review.add_argument(
        "--post", action="store_true",
        help="post the session's verdict (+external evidence) back to the server;"
        " needs --queue-id and asks before sending",
    )
    review.add_argument(
        "--yes", action="store_true",
        help="with --post: do not ask (you have read this item's session result)",
    )
    review.add_argument(
        "--override-thresholds",
        default="",
        metavar="REASON",
        help="approve past the §6.3 gate; the reason is recorded in the verdict note",
    )
    review.set_defaults(func=_cmd_review)

    relay = sub.add_parser(
        "relay",
        help="maintainer: shuttle the public mailbox's queue into the local one, publish the snapshot",
    )
    relay.add_argument(
        "--remote", default=None, help="the mailbox Worker's URL (default: the packaged one)"
    )
    relay.add_argument("--server-root", required=True, help="the local share-server data directory")
    mailbox_token = relay.add_mutually_exclusive_group(required=True)
    mailbox_token.add_argument(
        "--mailbox-token", help="the mailbox's maintainer token (a Worker secret)"
    )
    mailbox_token.add_argument(
        "--mailbox-token-file",
        help="read it from a file instead, so the secret stays out of shell history",
    )
    relay.add_argument(
        "--limit", type=int, default=None, help=f"queued items per batch (default {RELAY_BATCH_LIMIT})"
    )
    relay.add_argument(
        "--purge-days",
        type=int,
        default=None,
        help="retire queue items decided more than this many days ago (0 disables)",
    )
    relay.add_argument(
        "--publish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hand the mailbox the current snapshot when done",
    )
    relay.add_argument(
        "--signing-key-file",
        default=None,
        help="the maintainer's snapshot signing key (`share keygen`); required to publish",
    )
    relay.set_defaults(func=_cmd_relay)

    keygen = sub.add_parser(
        "keygen", help="maintainer: create the key that signs published snapshots"
    )
    keygen.add_argument("--out", required=True, help="where to write the private key (never overwritten)")
    keygen.set_defaults(func=_cmd_keygen)

    return parser.parse_args(argv)


#: Commands that write local state: they serialize behind the same
#: cross-process lock the maintenance CLI and auto-apply use — a pull racing
#: a knowledge update (or another pull) must not interleave. ``push`` is here
#: for the intent log: two concurrent pushes of the same content must agree
#: on one idempotency key, so the lookup/append (and, coarsely, the send)
#: happen under the lock.
_MUTATING = {"register", "mark", "unmark", "status", "pull", "push", "conflicts"}


def main(argv: list[str] | None = None) -> int:
    """Bind a terminal reporter for the whole command, like the sibling
    maintenance CLI does.

    Without it ``current_reporter()`` hands back the null reporter, whose
    ``warning()`` returns without doing anything — so every warning this CLI
    raises is dead code in production, and only visible to tests that bind a
    reporter of their own (reviewer, 2026-09-09).

    Only when nobody is already listening, though: an embedder that wrapped
    this call in its own ``reporting_to`` meant it, and taking the microphone
    away from them would be the same bug pointing the other way.
    """

    from finesub.reporting import (
        current_reporter,
        reporter_delivers,
        reporting_to,
        terminal_reporter,
    )

    if reporter_delivers(current_reporter()):
        return _main(argv)
    with reporting_to(terminal_reporter()):
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
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
