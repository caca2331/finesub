from __future__ import annotations

import pytest

from finesub.llm.knowledge.node.apply import apply_envelope, preview
from finesub.llm.knowledge.node.draft import Draft
from finesub.llm.knowledge.node.envelope import Binding, Envelope, EnvelopeError
from finesub.llm.knowledge.node.render import HandleMap, render_subject
from finesub.llm.knowledge.node.store import KnowledgeStore


def _seed(store: KnowledgeStore) -> tuple[str, str, str, str]:
    """subject S (common) with term T in section 角色 and misheard item I."""

    with store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "S", "intro": "i", "category": "common", "section_order": ["档案", "角色"]})
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "alias_text": "", "reading": "", "desc": "主角"})
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_item("I", "T", "misheard", "アリズ", fuzzy_enabled=True)
    return "S", "T", "M", "I"


def _bindings(store: KnowledgeStore, subject: str) -> list[Binding]:
    handles = HandleMap()
    render_subject(store, subject, mode="prompt", handles=handles)
    out = [Binding(**b) for b in handles.bindings()]
    for item in store.items_of("T"):
        out.append(Binding(handle="@i1", kind="item", id=item.item_id, expected_valid_from_rev=item.valid_from_rev))
    # membership handles are no longer rendered into the prompt (v79); ops that
    # target memberships get their bindings from the caller, so synthesize one
    for membership in store.children(subject):
        out.append(
            Binding(
                handle="@m1",
                kind="membership",
                id=membership.membership_id,
                expected_valid_from_rev=membership.valid_from_rev,
            )
        )
    return out


def _envelope(store: KnowledgeStore, ops: list[dict], *, rev: int | None = None, task_id: str = "t1") -> Envelope:
    bindings = _bindings(store, "S")
    return Envelope(
        task_id=task_id,
        assignment_id="a1",
        context_epoch=1,
        knowledge_read_rev=store.current_rev() if rev is None else rev,
        ops=ops,
        handle_bindings=bindings,
        draft_bindings=[op["handle"] for op in ops if op.get("op") == "create" and op.get("handle")],
    )


def test_batch_merges_ops_per_entity_and_supports_draft_handles(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(
            store,
            [
                {"op": "update", "id": "@k2", "set": {"payload.zh": "爱丽丝·A"}},
                {"op": "update", "id": "@k2", "set": {"payload.desc": "主角（改）"}},
                {"op": "create", "handle": "@new1", "kind": "term", "parent": "@k1", "section": "角色",
                 "payload": {"surface": "ボブ", "zh": "鲍勃", "alias_text": "", "reading": "", "desc": "配角"}},
                {"op": "add_item", "id": "@new1", "field": "misheard", "value": "ボフ"},
                {"op": "link", "id": "@new1", "rel": "see_also", "target": "@k2"},
            ],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back and result.conflicts == []
        assert result.applied_ops == [0, 1, 2, 3, 4]
        node = store.node("T")
        assert node.payload["zh"] == "爱丽丝·A" and node.payload["desc"] == "主角（改）"
        # two updates -> exactly one new version row
        rows = store.conn.execute("SELECT COUNT(*) AS n FROM node_versions WHERE local_id='T'").fetchone()["n"]
        assert rows == 2
        new_id = result.created["@new1"]
        assert store.node(new_id).payload["surface"] == "ボブ"
        assert [m.section for m in store.parents(new_id)] == ["角色"]
        assert [i.value for i in store.items_of(new_id)] == ["ボフ"]
        assert [l.target_id for l in store.links_from(new_id)] == ["T"]
        assert store.revision(result.rev).proposal_hash == env.proposal_hash()


def test_rejected_op_leaves_nothing_behind(tmp_path) -> None:
    """A parentless term create fails *after* touching the overlay: the op must
    be rejected atomically, not half-committed."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(
            store,
            [
                {"op": "create", "kind": "term", "payload": {"surface": "幽灵", "zh": "", "alias_text": "", "reading": "", "desc": "d"}},
                {"op": "update", "id": "@k2", "set": {"payload.zh": "爱丽丝·B"}},
            ],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back
        assert any("without a parent" in reason for _, reason in result.rejected_ops)
        assert store.node("T").payload["zh"] == "爱丽丝·B"
        assert all(n.payload.get("surface") != "幽灵" for n in store.nodes_of_kind("term"))


def test_concurrent_retire_of_a_read_only_dependency_drops_the_ops(tmp_path) -> None:
    """add_item's owner has no write op of its own; a concurrent retire of the
    owner must still fail the add instead of landing items on a dead node."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(store, [{"op": "add_item", "id": "@k2", "field": "misheard", "value": "アリサ"}])
        with store.begin("harness") as txn:
            txn.tombstone_membership("M")
            txn.tombstone_node("T")
        result = apply_envelope(store, env)
        assert not result.rolled_back
        assert any(c.entity == "item" and "retired concurrently" in c.reason for c in result.conflicts)
        assert all(item.value != "アリサ" for item in store.items_of("T"))


def test_retire_closes_items_and_links_and_restore_brings_them_back(tmp_path) -> None:
    from finesub.llm.knowledge.node.history import restore_node

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        with store.begin("user") as txn:
            txn.create_link("L", "T", "see_also", "S")
        env = _envelope(store, [{"op": "retire", "id": "@k2"}])
        result = apply_envelope(store, env)
        assert not result.rolled_back and result.conflicts == []
        assert store.node("T") is None
        assert store.items_of("T") == []
        assert store.links_from("T") == []
        assert store.parents("T") == []
        restore_node(store, "T")
        assert store.node("T") is not None
        assert [i.value for i in store.items_of("T")] == ["アリズ"]
        assert [l.target_id for l in store.links_from("T")] == ["S"]
        assert [m.section for m in store.parents("T")] == ["角色"]


def test_retire_plus_explicit_remove_item_is_idempotent(tmp_path) -> None:
    """One legal envelope may both remove an item and retire its owner; the
    cascade must not double-tombstone into a crash."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(
            store,
            [
                {"op": "remove_item", "item": "@i1"},
                {"op": "retire", "id": "@k2"},
            ],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back, result.rollback_reason
        assert store.node("T") is None
        assert store.items_of("T") == []


def test_stale_update_is_skipped_but_additive_ops_survive(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(
            store,
            [
                {"op": "update", "id": "@k2", "set": {"payload.desc": "来自旧视图"}},
                {"op": "add_item", "id": "@k2", "field": "aliases", "value": "Alice"},
                {"op": "add_item", "id": "@k2", "field": "misheard", "value": "アリズ"},  # already present -> rejected up front
            ],
        )
        # someone else updates T after the envelope was generated
        with store.begin("user") as txn:
            txn.update_node("T", payload={**store.node("T").payload, "desc": "并发改动"})
        result = apply_envelope(store, env)
        assert not result.rolled_back
        assert [c.entity for c in result.conflicts] == ["node"]
        assert "update skipped" in result.conflicts[0].reason
        assert store.node("T").payload["desc"] == "并发改动"
        assert "Alice" in [i.value for i in store.items_of("T")]
        assert result.applied_ops == [1]
        assert result.rejected_ops[0][0] == 2


def test_stale_retire_is_rejected_and_memberships_kept(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(store, [{"op": "retire", "id": "@k2", "merged_into": "@k1"}])
        with store.begin("user") as txn:
            txn.update_node("T", payload={**store.node("T").payload, "desc": "x"})
        result = apply_envelope(store, env)
        assert not result.rolled_back
        assert result.conflicts and "retire rejected" in result.conflicts[0].reason
        assert store.node("T") is not None
        assert store.children("S") and store.links_from("T") == []


def test_retire_tombstones_node_memberships_and_links_supersedes(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        before = store.current_rev()
        env = _envelope(store, [{"op": "retire", "id": "@k2", "merged_into": "@k1"}])
        result = apply_envelope(store, env)
        assert not result.rolled_back and result.rev == before + 1
        assert store.node("T") is None and store.children("S") == []
        assert store.node("T", before) is not None  # pinned read still sees it
        links = store.conn.execute("SELECT rel, target_id FROM link_versions WHERE source_id='T'").fetchall()
        assert [(r["rel"], r["target_id"]) for r in links] == [("supersedes", "S")]


def test_invalid_section_for_strict_preset_rolls_back_before_cas(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("S", "subject", {"surface": "主播", "intro": "", "category": "streamer"})
        env = Envelope(task_id="t", assignment_id="", context_epoch=0, knowledge_read_rev=store.current_rev(),
                       ops=[{"op": "create", "kind": "note", "parent": "S", "section": "自由节", "payload": {"text": "x"}}])
        result = apply_envelope(store, env)
        assert result.rolled_back and "not allowed by preset" in result.rollback_reason
        assert store.current_rev() == 1


def test_dependents_of_dropped_intent_are_dropped_then_rest_commits(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(
            store,
            [
                {"op": "remove_membership", "membership": "@m1"},  # stale after concurrent move
                {"op": "add_item", "id": "@k2", "field": "aliases", "value": "Ally"},
            ],
        )
        with store.begin("user") as txn:
            txn.move_membership("M", section="人名")
        result = apply_envelope(store, env)
        assert not result.rolled_back
        assert [c.entity for c in result.conflicts] == ["membership"]
        assert store.children("S")[0].section == "人名"
        assert "Ally" in [i.value for i in store.items_of("T")]


def test_preview_and_envelope_guards(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(store, [{"op": "update", "id": "@k9", "set": {"payload.desc": "x"}}])
        with pytest.raises(EnvelopeError):
            preview(store, env)
        good = _envelope(store, [{"op": "update", "id": "@k2", "set": {"payload.desc": "x"}}])
        overlay, problems = preview(store, good)
        assert problems == [] and overlay.nodes["T"].payload["desc"] == "x"
        assert store.current_rev() == 1  # preview has no side effects
        with pytest.raises(EnvelopeError):
            good.check_manifest(task_id="other", knowledge_read_rev=1)
        with pytest.raises(EnvelopeError):
            good.check_manifest(task_id="t1", knowledge_read_rev=0)
        good.check_manifest(task_id="t1", knowledge_read_rev=1, context_epoch=1)
        assert Envelope.from_dict(good.to_dict()).proposal_hash() == good.proposal_hash()
        assert good.validate_shape() == []
        bad = Envelope(task_id="t", assignment_id="", context_epoch=0, knowledge_read_rev=1,
                       ops=[{"op": "update", "id": "@k1"}, {"op": "nope"}])
        assert bad.validate_shape() == [
            "op[0] update: missing ['set']",
            "op[0] update: unbound handle @k1",
            "op[1]: unknown op 'nope'",
        ]


def test_set_cannot_touch_collections(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(store, [{"op": "update", "id": "@k2", "set": {"payload.misheard": ["x"]}}])
        result = apply_envelope(store, env)
        assert result.rejected_ops and "collections are item ops" in result.rejected_ops[0][1]
        # Every op rejected: nothing to write, so no revision is spent
        # (plan W2's no-empty-revision guard).
        assert result.rev is None and not result.rolled_back and result.applied_ops == []


def test_draft_dedupes_drops_resets_and_scopes_by_epoch() -> None:
    draft = Draft(task_id="t", context_epoch=1, knowledge_read_rev=3)
    draft.bind([Binding("@k1", "node", "S", 1)])
    op = {"op": "create", "handle": "@new1", "kind": "note", "parent": "@k1", "section": "档案", "payload": {"text": "x"}}
    first, dup = draft.propose(op)
    again, dup2 = draft.propose(dict(op))  # reconnect replay
    assert (first, dup, again, dup2) == ("d1", False, "d1", True)
    second, _ = draft.propose({"op": "add_item", "id": "@new1", "field": "aliases", "value": "y"})
    assert draft.status()["ops"][1]["draft_op_id"] == second
    with pytest.raises(EnvelopeError):
        draft.propose({"op": "create", "handle": "@new1", "kind": "note", "parent": "@k1", "section": "档案", "payload": {"text": "z"}})
    env = draft.to_envelope()
    assert env.draft_bindings == ["@new1"] and env.knowledge_read_rev == 3
    assert draft.drop(first) and "@new1" not in draft.draft_handles
    with pytest.raises(EnvelopeError):  # @new1 now unbound
        draft.to_envelope()
    restored = Draft.load(draft.to_dict(), task_id="t", context_epoch=1, knowledge_read_rev=3)
    assert [d.draft_op_id for d in restored.ops] == [second]
    assert Draft.load(draft.to_dict(), task_id="t", context_epoch=2, knowledge_read_rev=4).ops == []
    draft.reset()
    assert draft.status()["ops"] == [] and draft.handle_bindings == []


def test_create_inherits_share_visibility_from_the_shared_subject(tmp_path) -> None:
    """share_inherit at creation time (plan §6.4, review 2026-08-27): once a
    subject is marked shareable, new term nodes under it inherit. Facts do
    NOT (round 6): a new fact may be a real name — every fact needs its own
    explicit ``mark --kinds fact --match …``. A local subject inherits
    nothing."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        with store.begin("user") as txn:
            txn.update_node("S", visibility="shareable")
        env = _envelope(
            store,
            [
                {"op": "create", "handle": "@new1", "kind": "term", "parent": "@k1", "section": "角色",
                 "payload": {"surface": "ボブ", "zh": "鲍勃", "alias_text": "", "reading": "", "desc": "配角"}},
                {"op": "create", "handle": "@new2", "kind": "note", "parent": "@k1", "section": "档案",
                 "payload": {"label": "收录范围", "text": "只收近期篇章"}},
                {"op": "create", "handle": "@new3", "kind": "note", "parent": "@k1", "section": "档案",
                 "payload": {"label": "中之人", "text": "某某"}},
            ],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back
        # policy is (section, label), not kind (plan §3)
        assert store.node(result.created["@new1"]).visibility == "shareable"  # 分类节 inherits
        assert store.node(result.created["@new2"]).visibility == "local"      # label-level override
        assert store.node(result.created["@new3"]).visibility == "local"      # unregistered: fail-closed


def test_create_under_a_local_subject_stays_local(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        env = _envelope(
            store,
            [
                {"op": "create", "handle": "@new1", "kind": "term", "parent": "@k1", "section": "角色",
                 "payload": {"surface": "ボブ", "zh": "鲍勃", "alias_text": "", "reading": "", "desc": "配角"}},
            ],
        )
        result = apply_envelope(store, env)
        assert store.node(result.created["@new1"]).visibility == "local"


def test_moving_a_line_onto_a_taken_label_is_refused(tmp_path) -> None:
    """Review 2026-08-29 P1-1: a `move_membership` of a LOCAL line never loads
    the node, so a node-driven uniqueness check cannot see the move at all.
    The clash was visible at the envelope's read rev, so plan W2's per-op
    resolution stays out of it: an authoring error still rolls back whole."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("S", "subject", {"surface": "某主播", "intro": "i", "category": "streamer",
                                             "section_order": ["档案", "待归类"]})
            txn.create_node("A", "note", {"text": "2022-12", "label": "出道"})
            txn.create_node("B", "note", {"text": "2023-01", "label": "出道"})
            txn.create_membership("MA", "S", "A", "档案", 0)
            txn.create_membership("MB", "S", "B", "待归类", 0)
        rev = store.current_rev()
        env = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[{"op": "move_membership", "membership": "@m", "section": "档案"}],
            handle_bindings=[Binding(handle="@m", kind="membership", id="MB", expected_valid_from_rev=rev)],
        )
        result = apply_envelope(store, env)
        assert result.rolled_back and "重复" in result.rollback_reason
        assert [m.section for m in store.parents("B")] == ["待归类"]


def test_a_conflict_resolved_inside_the_txn_spends_no_revision(tmp_path, monkeypatch) -> None:
    """Plan W2: the competitor that commits AFTER our preview but BEFORE our
    BEGIN IMMEDIATE is resolved by the in-txn pass -- and when that empties the
    overlay, the revision `begin` opened must not commit with zero rows."""

    from finesub.llm.knowledge.node import apply as apply_mod

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("S", "subject", {"surface": "某主播", "intro": "i", "category": "streamer",
                                             "section_order": ["档案", "待归类"]})
            txn.create_node("A", "note", {"text": "2022-12", "label": "出道"})
            txn.create_membership("MA", "S", "A", "档案", 0)
        rev = store.current_rev()

        real_preview = apply_mod.preview
        raced: list[int] = []

        def racing_preview(st, env):
            out = real_preview(st, env)
            if not raced:
                raced.append(1)
                with st.begin("user") as txn:  # lands between preview and BEGIN
                    txn.create_node("C", "note", {"text": "x", "label": "新标记"})
                    txn.create_membership("MC", "S", "C", "档案", 1)
            return out

        monkeypatch.setattr(apply_mod, "preview", racing_preview)
        competitor_rev = rev + 1  # the racing preview commits exactly one
        env = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[{"op": "create", "handle": "@n", "kind": "note", "parent": "@s",
                  "section": "档案", "payload": {"text": "y", "label": "新标记"}}],
            handle_bindings=[Binding(handle="@s", kind="node", id="S", expected_valid_from_rev=rev)],
            draft_bindings=["@n"],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back
        assert result.rev is None  # the loser wrote nothing: no empty revision
        assert [c.entity for c in result.conflicts] == ["node", "membership"]
        assert store.current_rev() == competitor_rev  # only the winner's rev
        for table in ("node_versions", "item_versions", "membership_versions"):
            rows = store.conn.execute(
                f"SELECT COUNT(*) c FROM {table} WHERE valid_from_rev > ?", (competitor_rev,)
            ).fetchone()["c"]
            assert rows == 0


def test_a_concurrent_move_race_drops_only_the_move(tmp_path) -> None:
    """Plan W2: the same clash arising AFTER the envelope's read rev is a
    race, and the loser is the move OP alone, not the batch."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("S", "subject", {"surface": "某主播", "intro": "i", "category": "streamer",
                                             "section_order": ["档案", "待归类"]})
            txn.create_node("A", "note", {"text": "2022-12", "label": "出道"})
            txn.create_node("B", "note", {"text": "2023-01", "label": "出道"})
            txn.create_membership("MA", "S", "A", "待归类", 0)
            txn.create_membership("MB", "S", "B", "待归类", 1)
        rev = store.current_rev()
        with store.begin("user") as txn:  # a concurrent writer takes the label
            txn.create_node("C", "note", {"text": "x", "label": "出道"})
            txn.create_membership("MC", "S", "C", "档案", 0)
        env = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[{"op": "move_membership", "membership": "@m", "section": "档案"}],
            handle_bindings=[Binding(handle="@m", kind="membership", id="MB", expected_valid_from_rev=rev)],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back
        assert result.rev is None  # nothing left to write: no empty revision
        assert [c.entity for c in result.conflicts] == ["membership"]
        assert "已被占用" in result.conflicts[0].reason
        assert [m.section for m in store.parents("B")] == ["待归类"]
        assert [m.section for m in store.parents("C")] == ["档案"]  # winner kept


def test_a_shared_node_is_checked_in_every_section_it_sits_in(tmp_path) -> None:
    """Review 2026-08-29 P1-1: one node under two parents took whichever
    placement was walked first, so its other section went unchecked."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            for sid in ("S1", "S2"):
                txn.create_node(sid, "subject", {"surface": sid, "intro": "i", "category": "streamer",
                                                 "section_order": ["档案"]})
            txn.create_node("A", "note", {"text": "x", "label": "出道"}, visibility="shareable")
            txn.create_node("C", "note", {"text": "y", "label": "身高"})
            txn.create_membership("M1", "S1", "A", "档案", 0)
            txn.create_membership("M2", "S2", "A", "档案", 1)
            txn.create_membership("M3", "S2", "C", "档案", 2)
        rev = store.current_rev()
        env = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[{"op": "update", "id": "@a", "set": {"payload.label": "身高"}}],
            handle_bindings=[Binding(handle="@a", kind="node", id="A", expected_valid_from_rev=rev)],
        )
        result = apply_envelope(store, env)
        # [身高] was visible at the read rev: an authoring error, so the
        # whole-envelope rollback stays (W2 resolves only true races).
        assert result.rolled_back and "重复" in result.rollback_reason


def _labelled(store) -> int:
    """subject S (streamer) with two labelled notes in 档案; returns the rev."""

    with store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "某主播", "intro": "i", "category": "streamer",
                                         "section_order": ["档案"]})
        txn.create_node("A", "note", {"text": "a", "label": "甲"})
        txn.create_node("B", "note", {"text": "b", "label": "乙"})
        txn.create_membership("MA", "S", "A", "档案", 0)
        txn.create_membership("MB", "S", "B", "档案", 1)
    return store.current_rev()


def test_two_concurrent_relabels_cannot_both_land_the_same_label(tmp_path) -> None:
    """Review 2026-08-29 P1: the two batches touch DIFFERENT nodes, so the
    per-entity CAS never collides, and at the pinned read rev neither can see
    the other's line — the uniqueness check has to read the live store."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        rev = _labelled(store)

        def relabel(node_id: str) -> Envelope:
            return Envelope(
                task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
                ops=[{"op": "update", "id": "@n", "set": {"payload.label": "丙"}}],
                handle_bindings=[Binding(handle="@n", kind="node", id=node_id,
                                         expected_valid_from_rev=rev)],
            )

        first = apply_envelope(store, relabel("A"))
        second = apply_envelope(store, relabel("B"))   # same pinned rev = concurrent
        assert not first.rolled_back
        # Plan W2: the loser's relabel is reverted, not the whole envelope.
        assert not second.rolled_back and second.rev is None
        assert any("relabel reverted" in c.reason for c in second.conflicts)
        labels = [store.node(m.child_id).payload.get("label") for m in store.children("S")]
        assert labels.count("丙") == 1
        assert store.node("B").payload["label"] == "乙"


def test_a_node_gets_one_home_per_subject(tmp_path) -> None:
    """Review 2026-08-29 P2: the section membership check answered with a SET
    of child ids, so a second edge for a child already there collapsed and
    slipped through — the entry then rendered that line twice."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        rev = _labelled(store)
        again = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[{"op": "add_membership", "id": "@n", "parent": "@s", "section": "档案"}],
            handle_bindings=[Binding(handle="@n", kind="node", id="A", expected_valid_from_rev=rev),
                             Binding(handle="@s", kind="node", id="S", expected_valid_from_rev=rev)],
        )
        result = apply_envelope(store, again)
        assert result.rolled_back and "归属边" in result.rollback_reason
        assert [m.child_id for m in store.children("S")].count("A") == 1


def test_a_losing_relabel_keeps_the_envelopes_other_ops(tmp_path) -> None:
    """Plan W2: the uniqueness loser loses its OP, the rest of the envelope
    commits — a relabel onto a taken name is reverted to the stored label
    while the same op's other payload change and the batch's other ops land."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        rev = _labelled(store)
        with store.begin("user") as txn:  # a concurrent writer takes 丙 first
            txn.update_node("A", payload={"text": "a", "label": "丙"})
        env = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[
                {"op": "update", "id": "@n", "set": {"payload.label": "丙", "payload.text": "b2"}},
                {"op": "create", "handle": "@new1", "kind": "note", "parent": "@s",
                 "section": "档案", "payload": {"text": "c", "label": "丁"}},
            ],
            handle_bindings=[Binding(handle="@n", kind="node", id="B", expected_valid_from_rev=rev),
                             Binding(handle="@s", kind="node", id="S", expected_valid_from_rev=rev)],
            draft_bindings=["@new1"],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back and result.rev is not None
        assert any("relabel reverted" in c.reason for c in result.conflicts)
        node = store.node("B")
        assert node.payload["label"] == "乙"  # reverted to stored
        assert node.payload["text"] == "b2"  # the same op's other change landed
        assert store.node(result.created["@new1"]).payload["label"] == "丁"
        assert store.node("A").payload["label"] == "丙"  # the winner untouched


def test_a_losing_create_drops_only_its_closure(tmp_path) -> None:
    """Plan W2: a node created under a label another envelope took first
    (create-create race) goes with its dependency closure (membership,
    items, links) — and nothing else."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        rev = _labelled(store)
        with store.begin("user") as txn:  # the concurrent winner creates [己]
            txn.create_node("D", "note", {"text": "w", "label": "己"})
            txn.create_membership("MD", "S", "D", "档案", 5)
        env = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[
                {"op": "create", "handle": "@new1", "kind": "note", "parent": "@s",
                 "section": "档案", "payload": {"text": "x", "label": "己"}},  # just taken
                {"op": "add_item", "id": "@new1", "field": "aliases", "value": "someX"},
                {"op": "create", "handle": "@new2", "kind": "note", "parent": "@s",
                 "section": "档案", "payload": {"text": "y", "label": "戊"}},
            ],
            handle_bindings=[Binding(handle="@s", kind="node", id="S", expected_valid_from_rev=rev)],
            draft_bindings=["@new1", "@new2"],
        )
        result = apply_envelope(store, env)
        assert not result.rolled_back and result.rev is not None
        dropped_kinds = {c.entity for c in result.conflicts}
        assert "node" in dropped_kinds and "membership" in dropped_kinds and "item" in dropped_kinds
        labels = [store.node(m.child_id).payload.get("label") for m in store.children("S")]
        assert sorted(labels) == sorted(["甲", "乙", "己", "戊"])  # loser absent, sibling landed
        assert store.node("D").payload["text"] == "w"  # the winner's row untouched


def test_concurrent_appliers_queue_and_all_land(tmp_path) -> None:
    """Plan W2: in-process writers wait in the per-root single-writer queue
    instead of racing BEGIN IMMEDIATE — every envelope commits, one revision
    each, no busy-timeout losses."""

    import threading

    path = tmp_path / "kb.sqlite"
    with KnowledgeStore(path) as store:
        _labelled(store)
        before = store.current_rev()
    outcomes: list[tuple[int, bool]] = []
    lock = threading.Lock()
    # 30s, same as the joins: a cold interpreter can spend ten seconds on the
    # threads' first import + schema init and break the barrier spuriously.
    barrier = threading.Barrier(4, timeout=30)

    def writer(tag: int) -> None:
        with KnowledgeStore(path) as mine:  # SQLite: one connection per thread
            rev = mine.current_rev()
            env = Envelope(
                task_id=f"t{tag}", assignment_id="a", context_epoch=0,
                knowledge_read_rev=rev,
                ops=[{"op": "create", "kind": "subject", "payload": {
                    "surface": f"S{tag}", "intro": "i", "category": "common",
                    "section_order": []}}],
                handle_bindings=[],
            )
            barrier.wait()
            result = apply_envelope(mine, env)
            with lock:
                outcomes.append((result.rev or -1, result.rolled_back))

    threads = [threading.Thread(target=writer, args=(tag,)) for tag in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert all(not rolled for _rev, rolled in outcomes)
    assert sorted(rev for rev, _ in outcomes) == [before + 1, before + 2, before + 3, before + 4]


def test_one_node_may_still_live_under_two_subjects(tmp_path) -> None:
    """...and the sharing case stays supported: a different PARENT is a
    different home, not a duplicate."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        rev = _labelled(store)
        with store.begin("user") as txn:
            txn.create_node("S2", "subject", {"surface": "另一主播", "intro": "i",
                                              "category": "streamer", "section_order": ["档案"]})
        rev = store.current_rev()
        share = Envelope(
            task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=rev,
            ops=[{"op": "add_membership", "id": "@n", "parent": "@s2", "section": "档案"}],
            handle_bindings=[Binding(handle="@n", kind="node", id="A", expected_valid_from_rev=None),
                             Binding(handle="@s2", kind="node", id="S2", expected_valid_from_rev=None)],
        )
        result = apply_envelope(store, share)
        assert not result.rolled_back, result.rollback_reason
        assert {m.parent_id for m in store.parents("A")} == {"S", "S2"}


def test_provenance_booking_is_part_of_the_write_transaction(tmp_path, monkeypatch) -> None:
    """Booking ran after the commit, on an autocommit connection: it took the
    write lock once per row OUTSIDE the single-writer queue (observed as
    `database is locked` under concurrent applies), and a failure there raised
    past an already-committed revision. It belongs in the transaction."""

    from finesub.llm.knowledge.node import signals

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        before = store.current_rev()

        def boom(*args, **kwargs):
            raise RuntimeError("evidence write failed")

        monkeypatch.setattr(signals, "book_revision_evidence", boom)
        env = _envelope(store, [{"op": "update", "id": "@k2", "set": {"payload.zh": "爱丽丝·C"}}])
        with pytest.raises(RuntimeError, match="evidence write failed"):
            apply_envelope(store, env)
        # the revision went with it -- no committed write whose provenance failed
        assert store.current_rev() == before
        assert store.node("T").payload["zh"] == "爱丽丝"
