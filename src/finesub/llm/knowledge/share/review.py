"""Maintainer-side LLM review (plan §6.3): the half that never runs on the
server.

``python -m finesub.llm.knowledge.share review --remote … --maintainer-token …``
leases pending queue items, renders each into the review prompt
(``prompt_templates/share_review_v1.md``) and — **dry-run by default**, per
the repo-wide LLM convention — with ``--execute`` runs one review session per
item (native web search requested, so the agent form can produce per-claim
``source_refs``). The session answers ``approve | approve_tentative | reject``
(``REVIEW_VERDICTS``) plus an optional
``merge`` map (§6.2 ``merge_into``) and per-claim ``external_evidence``
(URL-backed corroboration, booked server-side as
``evidence(evidence_kind=external)`` on approval).

The verdict is still the maintainer's: without ``--post`` nothing is sent —
the CLI prints what the session concluded and the exact command to post it.
The deterministic slot-threshold pre-pass (§6.3 table) runs before any model
call and is part of the prompt, so the session knows which claims still need
corroboration; the LLM's own conclusion is never counted as a source.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ...output_tags import neutralize_closing_tags, parse_json_tag_block
from ...prompt_compose import load_prompt_template
from .exchange import threshold_report  # noqa: F401 - single truth, shared with the server gate

#: The verdict enum, single truth for the parser and the template guard test
#: (the schema block in ``share_review_v1.md`` must list exactly these — a
#: prose-only enum member is dead in practice: the model copies the schema).
REVIEW_VERDICTS = ("approve", "approve_tentative", "reject")

#: What an item claim registers, as the pending list names it: a misheard form is one no web page can confirm,
#: which the review prompt tells apart from an alias.
ITEM_FIELD_NAMES = {"aliases": "别名", "misheard": "误听"}

def _bundle_text(bundle: Mapping[str, Any]) -> str:
    """One line per entity. Free text goes through ``json.dumps`` so an
    embedded newline cannot fabricate structural lines in the prompt; the
    structural strings themselves are grammar-checked at server admission
    (``validate_bundle``, round 11) before any queue item reaches here."""

    def quoted(value: Any) -> str:
        return json.dumps(str(value or ""), ensure_ascii=False)

    lines: list[str] = []
    nodes = {str(node.get("handle")): node for node in bundle.get("nodes") or []}
    for handle, node in nodes.items():
        payload = json.dumps(node.get("payload") or {}, ensure_ascii=False, sort_keys=True)
        anchor = f" canonical={node.get('canonical_id')}" if node.get("canonical_id") else ""
        lines.append(f"{handle} [{node.get('kind')}]{anchor}: {payload}")
    for item in bundle.get("items") or []:
        lines.append(
            f"{item.get('handle')} [item {item.get('field')}] of {item.get('node')}:"
            f" {quoted(item.get('value'))}"
        )
    for membership in bundle.get("memberships") or []:
        lines.append(
            f"[membership] {membership.get('parent')} → {membership.get('child')}"
            f" ({quoted(membership.get('section'))})"
        )
    for link in bundle.get("links") or []:
        lines.append(f"[link] {link.get('source')} -{link.get('rel')}→ {link.get('target')}")
    return "\n".join(lines)


def _candidate_text(candidate: Mapping[str, Any] | None) -> str:
    """A hinted canonical node's content, quoted like the bundle lines: it
    came from a contributor once, so a newline in it must not fabricate rows."""

    if not candidate:
        return ""
    fields = {key: str(value) for key, value in candidate.items() if key != "kind" and value}
    return f" [{candidate.get('kind')}]: " + json.dumps(fields, ensure_ascii=False, sort_keys=True)


def pending_claims(item: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The claims the session is asked to corroborate, by a short per-session
    handle (``@v1``…). The verdict names a claim by its handle and the harness
    maps it back to ``(node, field_path, value_hash)``: copying a 64-hex hash
    and two path fields per row was a pure error surface (owner 2026-10-03,
    cold read Q14), and the hash is no use to the reader of the prompt."""

    rows = [row for row in threshold_report(item.get("bundle") or {}) if row["needs_external"]]
    return {f"@v{index + 1}": row for index, row in enumerate(rows)}


def render_review_prompt(item: Mapping[str, Any]) -> str:
    bundle = item.get("bundle") or {}
    item_fields = {f"items/{entry.get('handle')}": ITEM_FIELD_NAMES.get(str(entry.get("field") or ""), "")
                   for entry in bundle.get("items") or []}
    pending = []
    for handle, row in pending_claims(item).items():
        # label and self-reported sources are contributor text — quoted like
        # the bundle lines so an embedded newline cannot fabricate rows in
        # this trusted-looking section (round 12); node/field_path/slot are
        # grammar-checked at admission.
        notes = []
        if row.get("self_reported_refined"):
            notes.append("贡献者自述有精修印证——不可核实，仅参考")
        if row.get("self_reported_sources"):
            notes.append(
                "贡献者自述出处（不可核实，仅参考）："
                + "、".join(
                    json.dumps(str(ref), ensure_ascii=False)
                    for ref in row["self_reported_sources"][:3]
                )
            )
        label = json.dumps(str(row["label"]), ensure_ascii=False)
        field = item_fields.get(str(row["field_path"]), "")
        pending.append(
            f"- {handle}：{row['node']}（{label}）{row['field_path']}" + (f"（{field}）" if field else "")
            + f" [{row['slot']}]" + ("（" + "；".join(notes) + "）" if notes else "")
        )
    if not pending:
        pending = ["（无——所有 claim 已过门槛或属人工槽位）"]
    claims = [
        # The hash identifies the claim for the harness only; the reader names
        # claims by handle (`pending_claims`), so it is left out here too.
        json.dumps({k: v for k, v in claim.items() if k != "value_hash"}, ensure_ascii=False, sort_keys=True)
        for claim in bundle.get("claim_summaries") or []
    ]
    # No block at all without summaries: the gate section already says the
    # contributor's self-reports count for nothing (cold read round 4, P494).
    claim_summaries = (
        "<claim_summaries>\n" + neutralize_closing_tags("\n".join(claims)) + "\n</claim_summaries>\n\n"
        if claims else ""
    )
    # What each hinted node says comes in `merge_candidates`, a field of its
    # own beside `merge_hints`; a server older than that field sends ids only.
    known = item.get("merge_candidates") or {}
    hints = [
        f"- {handle} → {canonical}" + _candidate_text(known.get(canonical))
        for handle, candidates in (item.get("merge_hints") or {}).items()
        for canonical in candidates
    ] or ["（无）"]
    return load_prompt_template(
        "share_review_v1.md",
        strict=True,
        judgment=load_prompt_template("fragment_kb_judgment_v1.md", strict=True),
        # The contributor's text is quoted against fabricated lines and
        # neutralized against a closing tag that would end its block.
        pending_claims=neutralize_closing_tags("\n".join(pending)),
        claim_summaries=claim_summaries,
        merge_hints_text=neutralize_closing_tags("\n".join(hints)),
        bundle_text=neutralize_closing_tags(_bundle_text(bundle)),
    )


def parse_review_verdict(text: str, claims: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The session's ``<review_verdict>`` block, shape-checked; each
    ``external_evidence`` row's ``claim`` handle resolves through ``claims``
    (`pending_claims`). A row with an unknown handle is dropped and counted in
    ``unknown_claims`` -- like a row without a URL, it corroborates nothing."""

    data = parse_json_tag_block(text, "review_verdict")
    verdict = str(data.get("verdict") or "")
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(
            f"review verdict must be {'/'.join(REVIEW_VERDICTS)}, got {verdict!r}"
        )
    merge = data.get("merge") or {}
    if not isinstance(merge, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in merge.items()
    ):
        raise ValueError("merge must map bundle handles to canonical ids")
    evidence = data.get("external_evidence") or []
    if not isinstance(evidence, list):
        raise ValueError("external_evidence must be a list")
    rows = []
    unknown: list[str] = []
    for entry in evidence:
        if not isinstance(entry, Mapping) or not str(entry.get("url") or "").startswith("http"):
            continue  # a corroboration without a URL is not external evidence
        handle = str(entry.get("claim") or "")
        claim = claims.get(handle)
        if claim is None:
            unknown.append(handle or "(missing)")
            continue
        rows.append(
            {
                "node": str(claim["node"]),
                "field_path": str(claim["field_path"]),
                "value_hash": str(claim["value_hash"]),
                "url": str(entry.get("url")),
                "note": str(entry.get("note") or "")[:500],
            }
        )
    return {
        "verdict": verdict,
        "merge": dict(merge),
        "reason": str(data.get("reason") or "")[:500],
        "external_evidence": rows,
        "unknown_claims": unknown,
    }


class UnusableReviewReply(ValueError):
    """The session answered, but not with a verdict this module can read.

    Raised only from reading the reply -- a parse failure, or a reply cut off
    at its output ceiling -- so the review loop can skip that one item without
    also swallowing a configuration error on the way to the call."""


def run_review_session(item: Mapping[str, Any], *, client) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """One review call. Native web search is requested so the routed backend
    (agent form where bound) can actually corroborate claims; REST fallbacks
    simply return no external evidence, which the thresholds treat honestly
    (§6.3: REST 路径的审核只做 validator，不做外部印证)."""

    from ...client import OutputCutOff, reply_cut_off
    from ...routing.config import SESSION_OUTPUT_MAX_TOKENS, LLMRole

    prompt = render_review_prompt(item)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": prompt}],
        output_reserve=SESSION_OUTPUT_MAX_TOKENS,
        retrieval="native",
        task_group="research",
    )
    if reply_cut_off(result):
        raise UnusableReviewReply(str(OutputCutOff("review", result.requested_output_tokens)))
    try:
        return parse_review_verdict(result.content, pending_claims(item))
    except ValueError as exc:
        raise UnusableReviewReply(str(exc)) from exc
