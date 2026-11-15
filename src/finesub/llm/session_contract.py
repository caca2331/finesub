"""Declarative output contracts per LLM session + a shared validator.

Every text-block session's reply is a sequence of top-level sibling blocks. A
contract declares the mandatory blocks, split into two kinds:

- ``nonempty``: content-bearing blocks that must be present *and* non-empty
  (``context_pack`` …).
- ``present``: list-style blocks that must be present but may be an empty block
  (``analysis_notes``, ``search_queries``, ``keep_entries``, ``requested_entries`` …).

Blocks not listed are optional and unvalidated (e.g. a ``<reasoning>`` the model
writes on its own).

Only *top-level* siblings are declared here. Blocks that legitimately live one
level deeper — e.g. ``<void>`` inside ``<translated>`` — are the owning block's
internal concern, validated by that block's own parser (``output_protocol`` for the
correction round), and are never listed as a contract tag. Extraction is
nesting-aware (:func:`finesub.llm.output_tags.find_top_level_tag_blocks`), so a
mid-block name-drop can neither satisfy nor break a sibling requirement, and a
future deeper-nested tag needs no change here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .output_tags import find_top_level_tag_blocks


@dataclass(frozen=True)
class SessionContract:
    """The top-level output blocks a session's reply must contain."""

    nonempty: tuple[str, ...] = ()
    present: tuple[str, ...] = ()

    def validate(self, content: str) -> List[str]:
        """Return structural errors (empty list = pass)."""

        errors: List[str] = []
        for tag in self.nonempty:
            blocks = find_top_level_tag_blocks(content, tag)
            if not blocks:
                errors.append(f"missing <{tag}> block")
            elif not any(b.strip() for b in blocks):
                errors.append(f"empty <{tag}> block")
        for tag in self.present:
            if not find_top_level_tag_blocks(content, tag):
                errors.append(f"missing <{tag}> block")
        return errors


def prepare_task_contract(
    *, delivered: bool, emits_queries: bool = True, emits_entries: bool = True
) -> SessionContract:
    """The preparation task's two reply shapes (stage B, plan §2.1).

    One conversation, two possible replies: a *request* turn (analysis notes
    plus the request blocks the grant allows) or a *delivery* turn (the
    context pack). Which one a given reply is gets decided by the presence of
    ``context_pack``; each shape is then validated on its own.
    """

    if delivered:
        return SessionContract(nonempty=("context_pack",))
    # The prompt allows an empty `<analysis_notes>` ("没有值得记录的要点时输出空块").
    present: List[str] = ["analysis_notes"]
    if emits_entries:
        present.extend(("requested_entries", "keep_entries"))
    if emits_queries:
        present.append("search_queries")
    return SessionContract(present=tuple(present))


def correction_request_contract(*, search: bool) -> SessionContract:
    """The correction window's request turn: notes plus the block that marks a
    request under this grant -- the search block where harness search runs,
    else the entry request (the request turn's own shape gate,
    `stages.correction.request_turn.request_tags`, accepts either entry block)."""

    return SessionContract(
        present=("window_notes", "search_queries" if search else "requested_entries"),
    )


# Keyed by a stable session name shared by production stages and replay adapters.
# The correction (R2) round is intentionally absent: its reply is CSV, validated
# by output_protocol (schema, coverage, void handling), not by tag presence.
#
# The preparation entries are the ``retrieval=local`` shapes, derived from the
# same function production uses so the two can never drift. Replay fixtures are
# all frozen at ``local`` + ``continuity=serial``; the none/native and parallel
# shapes have no fixture yet (docs/llm_harness_behavior.md).
SESSION_CONTRACTS: Dict[str, SessionContract] = {
    "prepare_task_request": prepare_task_contract(delivered=False),
    "prepare_task_delivery": prepare_task_contract(delivered=True),
    # The correction window's request turn (task flow); its delivery shape is
    # the CSV round, validated by output_protocol like every correction reply.
    "correction_request": correction_request_contract(search=True),
}
