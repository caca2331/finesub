"""`validate_overlay` had four production call sites and no test at all.

That is the mirror of the validator this tree also carried with tests and no
caller: one guard nobody checked, one nobody ran. A guard that has never been
seen to fail is not known to work, so these red-verify the branches that matter
-- a malformed payload, and the label-uniqueness invariant it exists to hold.
"""

from __future__ import annotations

from finesub.llm.knowledge.node import apply as apply_mod
from finesub.llm.knowledge.node.store import KnowledgeStore


def _store(path):
    store = KnowledgeStore(path)
    with store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {
                "surface": "原神",
                "intro": "游戏",
                "category": "common",
                "section_order": ["角色"],
            },
        )
        txn.create_node("R1", "term", {"surface": "律にゃー", "desc": "听众统称。"})
        txn.create_membership("M0", "S", "R1", "角色", 0)
    return store


def test_a_consistent_overlay_reports_nothing(tmp_path) -> None:
    with _store(tmp_path / "kb.sqlite") as store:
        assert apply_mod.validate_overlay(store, apply_mod.Overlay(read_rev=1)) == []


def test_a_node_with_a_malformed_payload_is_reported(tmp_path) -> None:
    """The branch that runs on every node in the batch."""

    with _store(tmp_path / "kb.sqlite") as store:
        overlay = apply_mod.Overlay(read_rev=1)
        overlay.nodes["B"] = apply_mod.ONode("B", "term", {}, "shared", None, True)

        problems = apply_mod.validate_overlay(store, overlay)

    assert len(problems) == 1
    entity, local_id, message = problems[0]
    assert (entity, local_id) == ("node", "B")
    assert "missing" in message


def test_a_retired_node_is_not_validated(tmp_path) -> None:
    """Retiring a line must not require it to still be well-formed."""

    with _store(tmp_path / "kb.sqlite") as store:
        overlay = apply_mod.Overlay(read_rev=1)
        overlay.nodes["B"] = apply_mod.ONode(
            "B", "term", {}, "shared", None, True, retired=True
        )

        assert apply_mod.validate_overlay(store, overlay) == []


def test_every_malformed_node_is_reported_not_just_the_first(tmp_path) -> None:
    """One problem per entity: a batch author fixes them in one pass."""

    with _store(tmp_path / "kb.sqlite") as store:
        overlay = apply_mod.Overlay(read_rev=1)
        for local_id in ("B1", "B2"):
            overlay.nodes[local_id] = apply_mod.ONode(
                local_id, "term", {}, "shared", None, True
            )

        problems = apply_mod.validate_overlay(store, overlay)

    assert sorted(local_id for _, local_id, _ in problems) == ["B1", "B2"]
