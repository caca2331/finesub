"""The maintainer half of the mailbox deployment (plan B).

``deploy/share-worker/`` is a public mailbox that understands nothing: it takes
pushes, answers status by token, and serves whatever snapshot it was last
handed. Every judgement — sanitization, bundle admission, anchors, quotas, the
§6.3 evidence gate, the verdict itself — stays here, in the same
``ShareService`` the self-hosted deployment runs. This module is the shuttle
between the two, and it is deliberately the only thing that knows both:

1. **collect** — pull queued items and feed each to ``ShareService.push_as()``.
   Nothing is trusted on the way in: the local push path re-runs the whole
   admission boundary on a bundle that arrived over the network, which is
   exactly what it was written for.
2. **track** — re-ask the local queue about items the mailbox is still
   waiting on, so an approval that happens hours later reaches the
   contributor's ``share status``.
3. **publish** — hand the mailbox the current snapshot, so pulls see it.

Three properties worth keeping when this changes:

- **the relay is resumable and idempotent.** An item the mailbox already
  handed over is marked relayed; a crash between push and outbox leaves the
  item pending on both sides and the next run re-pushes it, where the local
  idempotency key collapses it onto the same queue item rather than forking one.
- **a rejected bundle is a normal outcome, not a failure.** The mailbox took
  it without judgement, so the first real verdict is often "this never should
  have been accepted" — that reaches the contributor as a rejection with the
  reason, and the relay carries on with the rest of the batch.
- **contributor identity survives the hop** (``adopt_contributor``): the
  digest counts distinct contributors, so collapsing them here would quietly
  turn every claim into one vote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from . import client
from .server import PURGE_SETTLED_DAYS, ShareError, ShareService

#: Terminal on the local queue: nothing the maintainer does later moves them.
FINAL_STATUSES = ("approved", "rejected", "expired")
#: Queue items fetched per request, in both the collect and the track pass.
RELAY_BATCH_LIMIT = 20


@dataclass
class RelayReport:
    collected: int = 0
    accepted: int = 0
    refused: int = 0
    tracked: int = 0
    advanced: int = 0
    published_rev: int | None = None
    purged: int = 0
    blanked: int = 0
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"collected {self.collected} queued push(es): {self.accepted} into the local queue,"
            f" {self.refused} refused at admission",
            f"tracked {self.tracked} open item(s), {self.advanced} reached a verdict",
        ]
        if self.purged or self.blanked:
            out.append(
                f"retired {self.purged} long-decided queue item(s) and blanked"
                f" {self.blanked} applied bundle(s)"
            )
        if self.published_rev is not None:
            out.append(f"published snapshot at server rev {self.published_rev}")
        return out + self.notes


def _verdict_result(
    service: ShareService, inbox_id: int, queue_id: int, report: RelayReport
) -> dict[str, Any]:
    """The full outcome of one decided queue item, as the mailbox stores it.

    ``assigned`` is the half that matters and the half that is easy to drop:
    it is the handle→canonical map the contributor's ``share status`` uses to
    backfill its own rows, and nothing regenerates it later.
    """

    try:
        status = service.push_status(queue_id, maintainer=service.maintainer_token)
    except ShareError as exc:
        report.notes.append(f"queue item {queue_id} (inbox {inbox_id}): {exc}")
        return {"inbox_id": inbox_id, "status": "pending", "queue_id": queue_id}
    return {
        "inbox_id": inbox_id,
        "status": status["status"],
        "verdict": status.get("verdict") or "",
        "assigned": status.get("assigned") or {},
        "queue_id": queue_id,
    }


def _collect(
    service: ShareService,
    remote: str,
    *,
    maintainer_token: str,
    limit: int,
    report: RelayReport,
) -> None:
    after = 0
    while True:
        batch = client.fetch_inbox(
            remote, maintainer_token=maintainer_token, after=after, limit=limit
        )
        items = list(batch.get("items") or [])
        if not items:
            return
        results: list[dict[str, Any]] = []
        for item in items:
            inbox_id = int(item.get("inbox_id") or 0)
            after = max(after, inbox_id)
            report.collected += 1
            bundle = item.get("bundle")
            contributor = str(item.get("contributor") or "")
            if not isinstance(bundle, Mapping) or not contributor:
                results.append(
                    {"inbox_id": inbox_id, "status": "rejected", "verdict": "malformed queue item"}
                )
                report.refused += 1
                continue
            try:
                reply = service.push_as(service.adopt_contributor(contributor), bundle)
            except ShareError as exc:
                # Admission refusals are the mailbox's whole point: it accepted
                # without looking, so this is where "no" gets said. The
                # contributor reads it through `share status`.
                results.append({"inbox_id": inbox_id, "status": "rejected", "verdict": str(exc)})
                report.refused += 1
                continue
            queue_id = int(reply.get("queue_id") or 0)
            status = str(reply.get("status") or "pending")
            if status in FINAL_STATUSES:
                # A re-push whose queue item was already decided: the local
                # push path answers with the idempotent duplicate, which
                # carries the status but NOT the verdict or the handle→
                # canonical map. Writing that back as-is would mark the item
                # terminal in the mailbox, and the tracking pass only looks at
                # pending ones — so the contributor's canonical ids would be
                # lost for good (reviewer, 2026-09-09). Ask for the whole
                # verdict before saying "done".
                results.append(_verdict_result(service, inbox_id, queue_id, report))
            else:
                results.append(
                    {"inbox_id": inbox_id, "status": status, "queue_id": queue_id}
                )
            report.accepted += 1
        client.post_outbox(remote, results, maintainer_token=maintainer_token)


def _track(
    service: ShareService,
    remote: str,
    *,
    maintainer_token: str,
    limit: int,
    report: RelayReport,
) -> None:
    """Ask the local queue about every item the mailbox is still waiting on.

    Paged, and the cursor is what makes it correct rather than merely tidy: the
    mailbox returns its open items oldest-first, so a single unpaged request
    would keep re-reading the same head of the queue. Items behind a backlog of
    still-undecided ones would never be asked about — an approval sitting at
    position 51 would simply never reach its contributor (reviewer, 2026-09-09).
    """

    after = 0
    while True:
        tracked = client.fetch_tracked(
            remote, maintainer_token=maintainer_token, after=after, limit=limit
        )
        items = list(tracked.get("items") or [])
        if not items:
            return
        results: list[dict[str, Any]] = []
        for item in items:
            inbox_id = int(item.get("inbox_id") or 0)
            queue_id = int(item.get("queue_id") or 0)
            after = max(after, inbox_id)
            report.tracked += 1
            try:
                status = service.push_status(queue_id, maintainer=service.maintainer_token)
            except ShareError as exc:
                report.notes.append(f"queue item {queue_id} (inbox {inbox_id}): {exc}")
                continue
            if status["status"] not in FINAL_STATUSES:
                continue
            results.append(_verdict_result(service, inbox_id, queue_id, report))
            report.advanced += 1
        if results:
            client.post_outbox(remote, results, maintainer_token=maintainer_token)


def run_relay(
    service: ShareService,
    remote: str,
    *,
    maintainer_token: str,
    limit: int | None = None,
    publish: bool = True,
    purge_days: int | None = None,
) -> RelayReport:
    """One full shuttle: collect, track, publish. Safe to re-run at any time."""

    limit = RELAY_BATCH_LIMIT if limit is None else limit
    report = RelayReport()
    _collect(service, remote, maintainer_token=maintainer_token, limit=limit, report=report)
    _track(service, remote, maintainer_token=maintainer_token, limit=limit, report=report)
    # After the tracking pass, never before: an item whose verdict has not
    # been carried back yet still has a reader.
    purged = service.purge_settled(
        PURGE_SETTLED_DAYS if purge_days is None else purge_days
    )
    report.purged, report.blanked = purged["deleted"], purged["blanked"]
    if publish:
        snapshot = service.snapshot()
        client.publish_snapshot(remote, snapshot, maintainer_token=maintainer_token)
        report.published_rev = int(snapshot.get("server_rev") or 0)
    return report
