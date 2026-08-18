"""Declarative output contracts per LLM session + a shared validator.

Every text-block session's reply is a sequence of top-level sibling blocks. A
contract declares the mandatory blocks, split into two kinds:

- ``nonempty``: content-bearing blocks that must be present *and* non-empty
  (``reasoning``, ``analysis_notes``, ``context_pack``, ``evidence_pack`` …).
- ``present``: list-style blocks that must be present but may be an empty block
  (``search_queries``, ``keep_entries``, ``requested_entries``, ``window_notes`` …).

Blocks not listed are optional and unvalidated (e.g. ``requested_entries`` in the
query round, which may be omitted entirely; the search-loop's ``search_queries``,
whose absence is the terminate signal).

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


def query_round_contract(*, knowledge_enabled: bool) -> SessionContract:
    """The per-window query round's contract.

    ``keep_entries`` is only demanded where the prompt asks for it: with no
    readable index the entry-request rules leave the prompt entirely, and a
    round must never be retried for omitting a block it was never told about.
    """

    present = ["window_notes"]
    if knowledge_enabled:
        present.append("keep_entries")
    present.append("search_queries")
    return SessionContract(nonempty=("reasoning",), present=tuple(present))


def fast_round1_contract(*, knowledge_enabled: bool) -> SessionContract:
    """Fast round 1's contract: the fused research+query round.

    The entry blocks are demanded only where the prompt shipped their rules --
    with no readable index (or ``--knowledge none``) they leave the prompt, so
    a correct reply legitimately omits them.
    """

    present = ["requested_entries", "keep_entries"] if knowledge_enabled else []
    present.append("search_queries")
    return SessionContract(
        nonempty=("reasoning", "analysis_notes"), present=tuple(present)
    )


def research_round1_contract(
    *, emits_queries: bool, emits_entries: bool, emits_notes: bool
) -> SessionContract:
    """R1's contract for one switch vector (docs/llm_harness_behavior.md).

    R1 is the "ask" half of the skeleton *r1 asks -> harness fetches -> r2
    digests*, and each half it asks for is owned by a different axis:
    ``search_queries`` by ``retrieval=local``, ``requested_entries`` /
    ``keep_entries`` by knowledge. ``analysis_notes`` is written *for r2*, so a
    run that skips r2 (``retrieval=none``) must not demand it -- nothing would
    read it. A round asked for neither half is not run at all.
    """

    nonempty = ["reasoning"]
    if emits_notes:
        nonempty.append("analysis_notes")
    present: List[str] = []
    if emits_entries:
        present.extend(("requested_entries", "keep_entries"))
    if emits_queries:
        present.append("search_queries")
    return SessionContract(nonempty=tuple(nonempty), present=tuple(present))


def research_round2_contract(*, emits_keep: bool) -> SessionContract:
    """R2's contract: always a context pack, ``keep_entries`` only with entries.

    R2 never emits ``search_queries`` in any vector -- under ``local`` the
    harness already fetched, and under ``native`` the model searched inside
    this very call.
    """

    return SessionContract(
        nonempty=("reasoning", "context_pack"),
        present=("keep_entries",) if emits_keep else (),
    )


# Keyed by a stable session name shared by production stages and replay adapters.
# The correction (R2) round is intentionally absent: its reply is CSV, validated
# by output_protocol (schema, singles, void handling), not by tag presence.
#
# The research entries are the ``retrieval=local`` shapes, derived from the same
# functions production uses so the two can never drift. Replay fixtures are all
# frozen at ``local`` + ``continuity=serial``; the none/native and parallel
# shapes have no fixture yet (docs/llm_harness_behavior.md).
SESSION_CONTRACTS: Dict[str, SessionContract] = {
    "query": query_round_contract(knowledge_enabled=True),
    "fast_round1": fast_round1_contract(knowledge_enabled=True),
    "research_round1": research_round1_contract(
        emits_queries=True, emits_entries=True, emits_notes=True
    ),
    "research_round2": research_round2_contract(emits_keep=True),
    "search_loop": SessionContract(
        nonempty=("reasoning", "evidence_pack"),
    ),
}
