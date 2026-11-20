"""The full share server's abuse bounds and secrets (security assessment
2026-09-25: F5 registration and snapshot cost, F6 the maintainer token, F9
contributor tokens at rest)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from finesub.llm.knowledge.share import server as server_module
from finesub.llm.knowledge.share.server import (
    ShareError,
    ShareService,
    contributor_id,
    serve,
)


def _service(tmp_path) -> ShareService:  # type: ignore[no-untyped-def]
    return ShareService(tmp_path / "srv", maintainer_token="mt")


def test_registrations_have_a_daily_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(server_module, "MAX_REGISTRATIONS_PER_DAY", 2)
    service = _service(tmp_path)
    service.register()
    service.register()
    with pytest.raises(ShareError, match="today"):
        service.register()


def test_an_unused_seat_is_given_back_but_an_approved_one_is_not(tmp_path) -> None:
    service = _service(tmp_path)
    idle = service.register()["token"]
    kept = service.register()["token"]
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    conn = service.store.conn
    conn.execute("UPDATE contributors SET created_at=?", (old,))
    for n, (token, status) in enumerate(((idle, "rejected"), (kept, "approved"))):
        conn.execute(
            "INSERT INTO share_queue(idempotency_key, contributor, bundle, received_at, status)"
            " VALUES (?, ?, '{}', ?, ?)",
            (f"k{n}", contributor_id(token), old, status),
        )
    service.register()  # registering is when seats are reclaimed
    left = {row[0] for row in conn.execute("SELECT token FROM contributors")}
    assert contributor_id(idle) not in left and contributor_id(kept) in left


def test_tokens_are_stored_hashed_and_old_stores_are_migrated(tmp_path) -> None:
    service = _service(tmp_path)
    token = service.register()["token"]
    stored = [row[0] for row in service.store.conn.execute("SELECT token FROM contributors")]
    assert stored == [contributor_id(token)] and token not in stored
    # a store from before: plaintext in both tables, no migration mark
    conn = service.store.conn
    conn.execute("UPDATE contributors SET token=?", (token,))
    conn.execute(
        "INSERT INTO share_queue(idempotency_key, contributor, bundle, received_at)"
        " VALUES ('k', ?, '{}', '2026-09-01T00:00:00+00:00')",
        (token,),
    )
    conn.execute("DELETE FROM meta WHERE key='share:contributor_ids'")
    service.close()
    reopened = _service(tmp_path)
    assert [r[0] for r in reopened.store.conn.execute("SELECT token FROM contributors")] == [
        contributor_id(token)
    ]
    assert reopened.push_status(1, token=token)["status"] == "pending"  # the holder still is who they were
    with pytest.raises(ShareError):
        reopened.push_status(1, token=contributor_id(token))  # the stored id is not a credential


def test_a_relayed_contributor_is_one_voter_across_the_worker_upgrade(tmp_path) -> None:
    """The mailbox used to relay the token itself and now relays its hash.
    A contributor approved before the upgrade (their row hashed by the store
    migration) and again after it must still count once in the digest tally."""

    service = _service(tmp_path)
    service.store.conn.execute(
        "INSERT INTO share_queue(idempotency_key, contributor, bundle, received_at, status)"
        " VALUES ('old', 'raw-worker-token', '', '2026-09-01T00:00:00+00:00', 'approved')"
    )
    service.store.conn.execute("DELETE FROM meta WHERE key='share:contributor_ids'")
    service.close()
    service = _service(tmp_path)  # the migration hashes the old row

    from_old_worker = service.adopt_contributor("raw-worker-token")
    from_new_worker = service.adopt_contributor(contributor_id("raw-worker-token"))
    voters = {row[0] for row in service.store.conn.execute("SELECT contributor FROM share_queue")}
    assert from_old_worker == from_new_worker == contributor_id("raw-worker-token")
    assert voters == {from_new_worker}


def test_the_migration_leaves_an_id_an_id(tmp_path) -> None:
    """An older relay stored what a new Worker sent -- already the hash -- and
    left the store unmarked; migrating must not hash it a second time."""

    service = _service(tmp_path)
    relayed = contributor_id("someone")
    service.store.conn.execute(
        "INSERT INTO share_queue(idempotency_key, contributor, bundle, received_at, status)"
        " VALUES ('k', ?, '', '2026-09-01T00:00:00+00:00', 'approved')",
        (relayed,),
    )
    service.store.conn.execute("DELETE FROM meta WHERE key='share:contributor_ids'")
    service.close()
    service = _service(tmp_path)
    assert [row[0] for row in service.store.conn.execute("SELECT contributor FROM share_queue")] == [relayed]


def test_a_corpus_every_client_would_refuse_is_not_published(tmp_path) -> None:
    """A row stored before a rule existed makes every pull fail admission; the
    maintainer hears it when building the snapshot, not from users later."""

    from finesub.llm.knowledge.node.store import KnowledgeStore

    root = tmp_path / "srv"
    root.mkdir()
    with KnowledgeStore(root / server_module.SERVER_STORE_FILENAME) as store:
        with store.begin("import") as txn:  # the store loads legacy rows unchecked
            txn.create_node("S", "subject", {"surface": "a|b", "intro": "", "category": "common"})
    service = ShareService(root, maintainer_token="mt")  # the chain starts at that revision
    with pytest.raises(ShareError, match="refused by every client"):
        service.snapshot()


def test_an_unchanged_corpus_is_snapshotted_once(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path)
    built = []
    real = server_module.snapshot_content
    monkeypatch.setattr(server_module, "snapshot_content", lambda *a, **k: built.append(1) or real(*a, **k))
    first = service.snapshot()
    assert service.snapshot() is first and len(built) == 1


def _get(url: str, **headers: str):  # type: ignore[no-untyped-def]
    request = urllib.request.Request(url)
    for key, value in headers.items():
        request.add_header(key, value)
    return urllib.request.urlopen(request, timeout=10)


def test_http_answers_304_to_the_current_head_and_refuses_foreign_hosts(tmp_path) -> None:
    import threading

    httpd = serve(tmp_path / "srv", port=0, maintainer_token="mt")
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        with _get(f"{base}/snapshot") as response:
            etag = response.headers["ETag"]
            assert json.loads(response.read())["chain_hash"] == etag.strip('"')
        with pytest.raises(urllib.error.HTTPError) as not_modified:
            _get(f"{base}/snapshot", **{"If-None-Match": etag})
        assert not_modified.value.code == 304
        with pytest.raises(urllib.error.HTTPError) as rebound:
            _get(f"{base}/identity", Host="evil.example")
        assert rebound.value.code == 421
    finally:
        httpd.shutdown()
        httpd.server_close()
        httpd.share_service.close()  # type: ignore[attr-defined]


def test_the_server_will_not_start_without_a_token_or_take_one_on_the_command_line(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(server_module.MAINTAINER_TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit):
        server_module.main(["--root", str(tmp_path / "srv")])
    with pytest.raises(SystemExit):
        server_module.main(["--root", str(tmp_path / "srv"), "--maintainer-token", "x"])
