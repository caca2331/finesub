"""The name check of a task that writes the knowledge base: the first step of its knowledge update stage
(docs/knowledge.md「中文定名的核查」).

The task's searching sessions -- the preparation task and each correction window -- searched the 未核 names they
wrote and recorded what they found (`feedback.NameCheckRecord`, the ``name_checks`` of their feedback). This step:

* **collects** them onto library lines (`collect_clues`): matched on the source word and the library name
  (`scan_normalize`d), at most `CLUES_PER_LINE` a line, a disagreeing spelling first. A record naming no live line,
  a 人定 one, or one the check may not send out (`name_check._checkable`) is dropped and counted. A search redirect
  link is followed once (`web_search.follow_search_redirect`, `follow_clue_links`), only on the lines that go and
  only once a route will take them;
* **chooses** the lines (`lines_to_send`): a line a clue spells otherwise goes unless a refutation of its current
  name still counts (`name_trust`: the correction was refused, and the line has not changed since); any other only
  when the check is owed it (`check_due`). A line disputed brings the lines whose name contains its name
  (`build_claims`), in its batch;
* **batches** them (`plan_batches`): at most `NATIVE_BATCH_LINES` lines a batch where the route searches on its own,
  `LOCAL_BATCH_LINES` where the harness searches for it (`name_check.name_check_route`); a line and those it brings
  together, an entry's lines together where they fit; batches run one after another, and a line that changed since
  the plan drops out of its batch;
* asks each batch (`name_check.run_name_check_session`, with its repair round) and lands it (`book_name_checks`) before
  the next. The session's own redirect links are followed before the write lock is taken.

A failure costs only the check (the subtitles are the product): no route, a session that fails, a ledger that cannot be
written are warnings, and the knowledge update runs as usual.

Ledger (`LEDGER_FILENAME`): each batch's claims, rows, replies, evidence task id (``run_id``, one per session: the same
clip checked again in another task directory a week later is a second confirmation) and whether it is booked, written
before anything lands. A resumed task first lands the batches answered but not booked from what they stored (same task
id: no evidence twice), then asks only the lines no stored batch answered; ``done`` marks a step that ran to its end,
which a resume replays without asking while the task's records are those it ran on (``records``); with other
records, or ``resume`` False, the stored batches still land and stay asked, and the rest is chosen again.
``apply`` False asks, reports and keeps nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from finesub.reporting import current_reporter
from finesub_bootstrap.fsops import write_atomic

from .feedback import NameCheckRecord, aggregate_task_update_feedback
from .name_check import (
    Claim,
    _checkable,
    book_name_checks,
    build_claims,
    landing_lines,
    name_check_route,
    pending_edits_hold,
    run_name_check_session,
)
from .node.matching import scan_normalize
from .node.name_trust import HUMAN, check_due, zh_name, zh_statuses
from .node.store import KnowledgeStore

LEDGER_FILENAME = "kb-name-check.json"
#: The task-artifact kind the task report reads (`task_report.py`).
CHECK_ARTIFACT = "kb_name_check"
#: Lines a batch holds, the lines sent along included: on a route with native search, and on the local path (the
#: harness searches; its material is budgeted for this many, `name_check.LOCAL_RESULT_TOKENS`).
NATIVE_BATCH_LINES = 30
LOCAL_BATCH_LINES = 8
#: Clues kept per line.
CLUES_PER_LINE = 3
#: Why a record was dropped, for the task report.
DROPPED_LABELS = {"no_line": "对不上库里的行", "human": "那一行是人定的", "not_sendable": "那一行不能送去核"}
UNPROCESSED = "这些行的分歧这次没处理，下一个带 --knowledge update 的任务会再搜，或跑 finesub knowledge verify"


# ---- the clues --------------------------------------------------------------------------------------------------


@dataclass
class Clues:
    #: node id -> the clues it keeps (`CLUES_PER_LINE`), a disagreeing spelling first
    by_line: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    #: node id -> how many records named it, before the cap
    counts: dict[str, int] = field(default_factory=dict)
    #: the lines some record spells otherwise than the library
    disputed: set[str] = field(default_factory=set)
    #: reason (`DROPPED_LABELS`) -> records dropped
    dropped: dict[str, int] = field(default_factory=dict)


def _key(source: str, zh: str) -> tuple[str, str]:
    return scan_normalize(source), scan_normalize(zh)


def collect_clues(store: KnowledgeStore, records: Sequence[NameCheckRecord], rev: int | None = None) -> Clues:
    """The records on the library's lines at ``rev``, their links as recorded (`follow_clue_links`)."""

    at = store.current_rev() if rev is None else rev
    by_key: dict[tuple[str, str], list[str]] = {}
    for node in store.nodes_of_kind("term", at):
        name = zh_name(node)
        if name is not None:
            by_key.setdefault(_key(str(node.payload.get("surface") or ""), name), []).append(node.local_id)
    checkable = {node.local_id: node for node in _checkable(store, at)}
    statuses = zh_statuses(store, checkable.values(), at)
    clues = Clues()
    raw: dict[str, list[NameCheckRecord]] = {}
    for record in records:
        ids = by_key.get(_key(record.source, record.kb_zh), [])
        sendable = [node_id for node_id in ids if node_id in checkable]
        usable = [node_id for node_id in sendable if statuses[node_id].state != HUMAN]
        if not usable:
            why = "no_line" if not ids else "human" if sendable else "not_sendable"
            clues.dropped[why] = clues.dropped.get(why, 0) + 1
            continue
        for node_id in usable:
            raw.setdefault(node_id, []).append(record)
    for node_id, found in raw.items():
        name = scan_normalize(zh_name(checkable[node_id]) or "")

        def disagrees(record: NameCheckRecord) -> bool:
            return bool(record.found) and scan_normalize(record.found) != name

        clues.counts[node_id] = len(found)
        if any(disagrees(record) for record in found):
            clues.disputed.add(node_id)
        # a disagreeing spelling first, then an agreeing one, then a search that found nothing; one per spelling and page
        ranked = sorted(found, key=lambda record: 0 if disagrees(record) else 1 if record.found else 2)
        kept: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for record in ranked:
            mark = (scan_normalize(record.found), record.url) if record.found else ("", record.query)
            if mark in seen:
                continue
            seen.add(mark)
            kept.append({key: value for key, value in (
                ("found", record.found), ("url", record.url), ("quote", record.quote), ("query", record.query),
                ("origin", record.origin)) if value})
            if len(kept) == CLUES_PER_LINE:
                break
        clues.by_line[node_id] = kept
    return clues


def follow_clue_links(claims: Sequence[Claim], *, follow: Callable[[str], str] | None = None) -> list[Claim]:
    """The claims with each clue's search redirect link turned into the page's address (default
    `web_search.follow_search_redirect`, once per link), so the session and the evidence see the page itself."""

    if follow is None:
        from ..web_search import follow_search_redirect as follow
    followed: dict[str, str] = {}

    def page(url: str) -> str:
        if url not in followed:
            followed[url] = follow(url)
        return followed[url]

    return [replace(claim, clues=[{**clue, "url": page(clue["url"])} if clue.get("url") else dict(clue)
                                  for clue in claim.clues]) if claim.clues else claim for claim in claims]


def lines_to_send(store: KnowledgeStore, clues: Clues, rev: int | None = None, *, skip: Sequence[str] = (),
                  now: datetime | None = None) -> list[str]:
    """The lines the check takes, disputed first, then the most recorded: a disputed line unless a refutation of its
    current name still counts, any other only when `check_due` says the check is owed it; none in ``skip``."""

    at = store.current_rev() if rev is None else rev
    nodes = [node for node_id in clues.by_line if node_id not in skip
             for node in [store.node(node_id, at)] if node is not None]
    statuses = zh_statuses(store, nodes, at)
    order = {node_id: index for index, node_id in enumerate(clues.by_line)}
    chosen = [
        node.local_id for node in nodes
        if (not statuses[node.local_id].refuted if node.local_id in clues.disputed
            else check_due(statuses[node.local_id], now))
    ]
    return sorted(chosen, key=lambda node_id: (node_id not in clues.disputed, -clues.counts.get(node_id, 0),
                                               order[node_id]))


def plan_batches(claims: Sequence[Claim], size: int) -> list[list[Claim]]:
    """The claims in batches of at most ``size``, an entry's lines together, in the claims' order. A line sent along
    (``along``) goes in its parent's batch, through any chain: it is answered only beside the line it follows
    (`_current` drops it otherwise). A line sent for itself is its own group even when its name contains another
    sent line's (its ``derived_from`` is then only a hint). A group is one line and the lines it brings; past
    ``size`` it keeps its first ``size`` lines (the line first) and the rest do not go this time, as the lines past
    `DERIVED_MAX_LINES` never do: the batch size holds, and a line not brought is checked when it falls due itself."""

    by_id = {claim.claim_id: claim for claim in claims}

    def root(claim: Claim) -> str:
        seen = set()
        while claim.along and claim.derived_from in by_id and claim.claim_id not in seen:
            seen.add(claim.claim_id)
            claim = by_id[claim.derived_from]
        return claim.claim_id

    groups: dict[str, list[Claim]] = {}
    for claim in claims:
        groups.setdefault(root(claim), []).append(claim)
    subject_rank: dict[str, int] = {}
    for claim in claims:
        subject_rank.setdefault(claim.subject, len(subject_rank))
    ordered = sorted(groups.values(), key=lambda group: subject_rank[by_id[root(group[0])].subject])
    batches: list[list[Claim]] = []
    current: list[Claim] = []
    for group in ordered:
        group = group[:size]
        if current and len(current) + len(group) > size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def still_sendable(store: KnowledgeStore, planned: Sequence[Claim]) -> list[Claim]:
    """The planned claims that may still go out, ids kept: a line retired, changed or no longer one the check may
    send out (moved into a section that may not be, `name_check._checkable`) since the plan dropped, then a line
    sent along whose parent dropped. The task's step and `knowledge verify` both check a batch with it as it goes."""

    sendable = {node.local_id: node for node in _checkable(store, store.current_rev())}

    def unchanged(claim: Claim) -> bool:
        node = sendable.get(claim.node_id)
        return node is not None and (str(node.payload.get("surface") or ""), zh_name(node) or "",
                                     str(node.payload.get("desc") or "")) == (claim.surface, claim.zh, claim.desc)

    kept = [claim for claim in planned if unchanged(claim)]
    while True:
        present = {claim.claim_id for claim in kept}
        trimmed = [claim for claim in kept if not claim.along or claim.derived_from in present]
        if len(trimmed) == len(kept):
            # a line sent for itself may name a line in another batch as its hint: the prompt names only this one's
            return [claim if claim.derived_from in present or claim.derived_from is None
                    else replace(claim, derived_from=None) for claim in kept]
        kept = trimmed


def _current(store: KnowledgeStore, planned: Sequence[Claim]) -> list[Claim]:
    """The batch as it goes out (`still_sendable`), ids numbered afresh."""

    kept = still_sendable(store, planned)
    ids = {claim.claim_id: f"n{index + 1}" for index, claim in enumerate(kept)}
    return [replace(claim, claim_id=ids[claim.claim_id], derived_from=ids.get(claim.derived_from or ""))
            for claim in kept]


# ---- the step ---------------------------------------------------------------------------------------------------


def run_name_check_step(
    *,
    knowledge_root: Any,
    artifact_dir: Path,
    task_id: str,
    client: Any,
    apply: bool = True,
    retries: int | None = None,
    records: Sequence[NameCheckRecord] | None = None,
    search: Any = None,
    resume: bool = True,
) -> dict[str, Any] | None:
    """The check over this task's records (None: read from ``artifact_dir``'s feedback); returns what it did, None
    when there was nothing to check. ``search``: the local path's search client (`name_check.name_check_route`).
    Never raises for a by-product."""

    from .base import append_task_artifact, knowledge_root_path
    from .node.repo import KnowledgeRepo

    artifact_dir = Path(artifact_dir)
    ledger_path = artifact_dir / LEDGER_FILENAME
    ledger = _read_ledger(ledger_path) if apply else {}
    if records is None:
        records = aggregate_task_update_feedback([artifact_dir]).all_name_checks()
    digest = _records_digest(records)
    if resume and ledger.get("done") and ledger.get("records") == digest:
        return {"replayed": True, **(ledger.get("summary") or {})}
    batches: list[dict[str, Any]] = [dict(batch) for batch in ledger.get("batches") or ()]
    if not records and not batches:
        return None
    root = knowledge_root_path(knowledge_root)
    repo = KnowledgeRepo.open(root)
    summary: dict[str, Any] = {"lines": 0, "sent": [], "batches": [], "dropped": {}}

    def keep() -> bool:
        if not apply:
            return True
        try:
            _write_ledger(ledger_path, {"task_id": task_id, "batches": batches})
        except OSError as exc:  # landing without a kept answer could leave a written base no resume can account for
            current_reporter().warning("kb-name-check-failed", f"中文定名核查的答案没能存下：{exc}", impact=UNPROCESSED)
            summary["failed"] = f"ledger: {exc}"
            return False
        return True

    for batch in batches:  # answered by an interrupted run: land what it stored, ask nothing again
        if batch.get("status") == "answered":
            _land(repo, root, batch)
            keep()
    asked = {claim["node_id"] for batch in batches for claim in batch.get("claims") or ()
             if claim["claim_id"] in {row["claim_id"] for row in batch.get("rows") or ()}}
    clues = collect_clues(repo.store, records)
    summary["dropped"] = dict(clues.dropped)
    send = lines_to_send(repo.store, clues, skip=sorted(asked))
    claims = build_claims(repo.store, send, clues=clues.by_line, derive_for=clues.disputed, skip=asked)
    route, why = name_check_route(client, search=search) if claims else (None, "")
    if route is not None:
        claims = follow_clue_links(claims)
    summary["lines"] = len(claims)
    summary["sent"] = [{"line": f"{claim.surface}|{claim.zh}", "subject": claim.subject, "clues": claim.clues}
                       for claim in claims if not claim.along]
    planned: list[list[Claim]] = []
    if route is not None:
        summary["route"] = route.kind
        planned = plan_batches(claims, NATIVE_BATCH_LINES if route.kind == "native" else LOCAL_BATCH_LINES)
    elif claims:
        current_reporter().note("kb-name-check-unavailable",
                                f"中文定名核查没有可用的路由（research 组的 GENERAL_CAPABLE）：{why}；{UNPROCESSED}")
        summary["skipped"] = "no_route"
    for plan in planned:
        claims_out = _current(repo.store, plan)
        if not claims_out:
            continue
        try:
            session = run_name_check_session(claims_out, client=client, retries=retries, route=route)
        except Exception as exc:  # the subtitles are the product; the knowledge base is a by-product
            current_reporter().warning("kb-name-check-failed", f"中文定名核查会话没有交出答案：{exc}",
                                       impact=UNPROCESSED)
            summary["failed"] = str(exc)
            break
        batch = {
            "status": "answered",
            # each session its own evidence task id (`name_trust` counts one confirmation per task id)
            "run_id": f"{task_id}-zh-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:6]}",
            "claims": [claim.to_dict() for claim in claims_out],
            "rows": [_followed(row) for row in session.rows],
            "answers": session.answers,
            "problems": session.problems,
            **({"search": session.search} if session.search else {}),
        }
        batches.append(batch)
        if not apply:
            batch["status"] = "not_applied"
            continue
        if not keep():
            break
        _land(repo, root, batch)
        if not keep():
            break
    summary["batches"] = [{key: batch[key] for key in ("status", "run_id", "report", "problems", "search")
                           if key in batch}
                          | {"lines": len(batch.get("claims") or ())} for batch in batches]
    if not apply:
        summary["rows"] = [row for batch in batches for row in batch.get("rows") or ()]
    elif (batches and not summary.get("failed") and not summary.get("skipped")
          and all(batch.get("status") == "booked" for batch in batches)):
        # done only once every batch landed and nothing was left unasked: a batch left to retry lands on the next
        # run, and lines a missing route left go when a route is back
        try:
            _write_ledger(ledger_path, {"task_id": task_id, "batches": batches, "done": True, "records": digest,
                                        "summary": summary})
        except OSError:
            pass  # the batches are kept already; a resume replays the landing and asks nothing new
    append_task_artifact(artifact_dir, kind=CHECK_ARTIFACT, task_id=task_id, payload=summary)
    current_reporter().debug("kb-name-check", {
        "lines": [line for batch in batches for line in landing_lines(batch.get("report") or {})]})
    return summary


def _followed(row: Mapping[str, Any]) -> dict[str, Any]:
    """A row with the session's redirect link replaced by the page's address, before any lock is taken."""

    from ..web_search import follow_search_redirect

    return {**row, "url": follow_search_redirect(row["url"])} if row.get("url") else dict(row)


def _land(repo: Any, root: Path, batch: dict[str, Any]) -> None:
    """Book one batch; it becomes ``booked`` unless the lock was busy or a correction waits (``retry``)."""

    from .base import knowledge_write_lock

    claims = [Claim.from_dict(item) for item in batch["claims"]]
    try:
        with knowledge_write_lock(root) as acquired:
            landing = book_name_checks(repo.store, batch["rows"], claims, task_id=batch["run_id"], write=acquired,
                                       hold=pending_edits_hold(repo))
            if acquired:
                repo.refresh_rendered()  # the check notes of rendered/ follow the evidence, not only a revision
    except Exception as exc:  # the subtitles are the product; the knowledge base is a by-product
        current_reporter().warning("kb-name-check-failed", f"中文定名核查的结论没能落库：{exc}",
                                   impact="答案留在任务目录里，这个任务下次续跑时再落")
        return
    report = landing.to_dict()
    batch["report"] = report
    if not acquired:
        current_reporter().warning("knowledge-locked", "另一个 FineSub 进程正在写知识库，中文定名核查的修正这次没落库",
                                   impact="证据已记，修正留到这个任务下次续跑时再落")
    elif not any(row.get("retry") for row in report.get("blocked", ())):
        batch["status"] = "booked"


def _read_ledger(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("batches", []), list) else {}


def _write_ledger(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))


def _records_digest(records: Sequence[NameCheckRecord]) -> str:
    """What a finished ledger was computed from: a step marked done replays only while the records are these."""

    from dataclasses import asdict
    from hashlib import sha256

    return sha256(json.dumps([asdict(record) for record in records], ensure_ascii=False, sort_keys=True)
                  .encode("utf-8")).hexdigest()
