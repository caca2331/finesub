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

from ...output_tags import parse_json_tag_block
from ...prompt_compose import load_prompt_template
from .exchange import threshold_report  # noqa: F401 - single truth, shared with the server gate

REVIEW_MAX_TOKENS = 8_192

#: The verdict enum, single truth for the parser and the template guard test
#: (the schema block in ``share_review_v1.md`` must list exactly these — a
#: prose-only enum member is dead in practice: the model copies the schema).
REVIEW_VERDICTS = ("approve", "approve_tentative", "reject")

THRESHOLD_TABLE = """- term（定名/别名/误听）：独立来源 ≥ 2，或一次精修印证，或带出处的外部印证
- streamer 档案 fact：带出处的外部印证（改本名/人设影响全库；精修不放行）
- 人际关系 / 重要经历：独立来源 ≥ 2 或精修印证（外部出处不单独放行）
- prose（简介/说话风格/正文）：人工判断（你的裁定即人工判断）
注意：贡献者随包自述的精修印证/出处**都不可核实、不作数**——服务端只认你在
external_evidence 里给出的、你亲自核实过的 URL；精修/独立来源两路要等服务端能
自行核实（跨用户聚合）才生效，在那之前这两路的批准都要维护者显式 override。"""


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


def render_review_prompt(item: Mapping[str, Any]) -> str:
    bundle = item.get("bundle") or {}
    thresholds = threshold_report(bundle)
    pending = []
    for row in thresholds:
        if not row["needs_external"]:
            continue
        # label and self-reported sources are contributor text — quoted like
        # the bundle lines so an embedded newline cannot fabricate rows in
        # this trusted-looking section (round 12); node/field_path/value_hash/
        # slot are grammar-checked at admission.
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
        pending.append(
            f"- {row['node']}（{label}）{row['field_path']} value_hash={row['value_hash']}"
            f" [{row['slot']}]" + ("（" + "；".join(notes) + "）" if notes else "")
        )
    if not pending:
        pending = ["（无——所有 claim 已过门槛或属人工槽位）"]
    claims = [
        json.dumps(claim, ensure_ascii=False, sort_keys=True)
        for claim in bundle.get("claim_summaries") or []
    ] or ["（无 claim 摘要——注意：claim 摘要是贡献者自述，不作为任何门槛依据）"]
    hints = [
        f"- {handle} 可能与既有节点相同：{', '.join(candidates)}"
        for handle, candidates in (item.get("merge_hints") or {}).items()
    ] or ["（无）"]
    return load_prompt_template(
        "share_review_v1.md",
        strict=True,
        judgment=load_prompt_template("fragment_kb_judgment_v1.md", strict=True),
        threshold_table=THRESHOLD_TABLE,
        pending_claims="\n".join(pending),
        claims_text="\n".join(claims),
        merge_hints_text="\n".join(hints),
        bundle_text=_bundle_text(bundle),
    )


def parse_review_verdict(text: str) -> dict[str, Any]:
    """The session's ``<review_verdict>`` block, shape-checked."""

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
    for entry in evidence:
        if not isinstance(entry, Mapping) or not str(entry.get("url") or "").startswith("http"):
            continue  # a corroboration without a URL is not external evidence
        rows.append(
            {
                "node": str(entry.get("node") or ""),
                "field_path": str(entry.get("field_path") or ""),
                "value_hash": str(entry.get("value_hash") or ""),
                "url": str(entry.get("url")),
                "note": str(entry.get("note") or "")[:500],
            }
        )
    return {
        "verdict": verdict,
        "merge": dict(merge),
        "reason": str(data.get("reason") or "")[:500],
        "external_evidence": rows,
    }


def run_review_session(item: Mapping[str, Any], *, client) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """One review call. Native web search is requested so the routed backend
    (agent form where bound) can actually corroborate claims; REST fallbacks
    simply return no external evidence, which the thresholds treat honestly
    (§6.3: REST 路径的审核只做 validator，不做外部印证)."""

    from ...routing.config import LLMRole

    prompt = render_review_prompt(item)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": prompt}],
        max_tokens=REVIEW_MAX_TOKENS,
        retrieval="native",
        task_group="research",
    )
    return parse_review_verdict(result.content)
