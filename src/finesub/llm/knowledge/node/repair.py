"""KB repair task (plan §11.3 / O3): the LLM half of the second migration,
and the general "arbitrary text → this library's standard ops" converter.

Two input modes, one output contract:

* candidates mode — the ``scan`` pass's judgment items (kind
  reclassification, grab-bag splits, episodic descs) grouped per subject;
* material mode — a user-supplied text handed to one subject.

The session sees the subject's PROMPT projection (with ``@k`` handles) and
answers in the existing ``<knowledge_proposals>`` block; application goes
through ``apply_model_proposals`` — the same validate→apply single path as
every other write (O16), nothing here holds write authority. Dry-run by
default, like every LLM entry point in this repo."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from ...prompt_compose import load_prompt_template, reasoning_clause
from .render import HandleMap
from .repo import KnowledgeRepo
from .scan import scan_candidates

REPAIR_MAX_TOKENS = 8_192

#: candidate kinds from ``scan`` the repair session can act on. chapter-alias
#: joins the flow (plan A6) but is verdict-only: the prompt forbids proposing
#: the deletion itself — the session may dismiss or flag needs_human.
#: `relation-review` / `grab-bag-fact` retired with the kinds they named;
#: `staging-line` is their successor — Phase B parks whatever needs a
#: judgement call in the staging section instead of guessing.
_REPAIRABLE = (
    "staging-line", "unnamed-term", "episodic-desc", "duplicate-term", "chapter-alias",
)

#: candidate verdict enum — single truth for the prompt text (rendered from
#: this tuple) and the parser; needs_human is the explicit "cannot decide,
#: cannot verify" exit that lands in the candidate ledger (plan A6).
CANDIDATE_VERDICTS = ("propose", "dismiss", "needs_human")

_VERDICTS_RE = re.compile(
    r"<candidate_verdicts\b[^>]*>(?P<body>.*?)</candidate_verdicts>",
    re.IGNORECASE | re.DOTALL,
)


def repair_targets(repo: KnowledgeRepo) -> dict[str, list[dict[str, Any]]]:
    """Second-pass judgment candidates grouped by owning subject id.

    Candidates with a standing ledger decision on their current content
    (dismissed / applied / pending_human) are filtered out — the scan no
    longer re-surfaces settled questions every round (plan A6). Routing is
    explicit: every candidate carries ``subject_id`` from the scan; one
    without it is a scan bug and is surfaced, never guessed."""

    from finesub.reporting import current_reporter

    from .candidates import filter_undecided

    store = repo.store
    rev = repo.rev
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in filter_undecided(store, scan_candidates(store, rev).candidates):
        if candidate.get("kind") not in _REPAIRABLE:
            continue
        subject_id = str(candidate.get("subject_id") or "")
        if not subject_id:
            current_reporter().warning(
                "knowledge-candidate-unrouted",
                f"扫描候选缺 subject_id，无法路由到条目：{dict(candidate)}",
            )
            continue
        grouped.setdefault(subject_id, []).append(candidate)
    return grouped


def candidate_handle_map(candidates: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Stable per-session handles (``@c1``…) the verdict block refers to."""

    return {f"@c{index + 1}": candidate for index, candidate in enumerate(candidates)}


def parse_candidate_verdicts(
    text: str, valid_handles: set[str]
) -> list[dict[str, str]]:
    """Rows of the session's ``<candidate_verdicts>`` block; rows with an
    unknown handle or verdict are dropped (reported by the caller's booking
    summary, not fatal — the candidate just stays open)."""

    from .proposals import strip_reasoning

    # same hijack as the proposals block: a model that names the tag while
    # reasoning would otherwise open the match inside <reasoning>
    match = _VERDICTS_RE.search(strip_reasoning(text))
    if not match:
        return []
    rows: list[dict[str, str]] = []
    for line in match.group("body").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        handle = str(data.get("candidate", ""))
        verdict = str(data.get("verdict", ""))
        if handle not in valid_handles or verdict not in CANDIDATE_VERDICTS:
            continue

        def _text(key: str) -> str:
            # Only a real JSON string counts: models emit nullable fields, and
            # ``str(None)`` is the non-empty "None" — which walked straight
            # past the needs_human evidence requirement (review 2026-08-28
            # P2-1 post-merge). null / arrays / objects read as empty.
            value = data.get(key)
            return value.strip()[:500] if isinstance(value, str) else ""

        rows.append({"candidate": handle, "verdict": verdict,
                     "reason": _text("reason"), "missing": _text("missing")})
    return rows


def render_repair_prompt(
    repo: KnowledgeRepo,
    subject_id: str,
    *,
    candidates: list[Mapping[str, Any]] | None = None,
    material: str = "",
    user_prompt: str = "",
    handles: HandleMap | None = None,
    candidate_handles: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    handles = handles if handles is not None else HandleMap()
    entry_text = repo.entry_prompt_text(subject_id, handles)
    if material:
        # Both task sections are templates now, like every other prompt text
        # here -- they were the one place the "prompt text is never hardcoded
        # in Python" rule was still broken.
        steer = user_prompt.strip()
        task = load_prompt_template(
            "fragment_kb_task_material_v1.md",
            strict=True,
            material=material.strip(),
            user_prompt_block=(
                "用户另有交代（它决定你**偏向**收什么、怎么写，"
                "但不能越过上面的收录标准）：\n\n" + steer
                if steer
                else "（用户没有额外交代。）"
            ),
        )
    else:
        cmap = (
            dict(candidate_handles)
            if candidate_handles is not None
            else candidate_handle_map(list(candidates or []))
        )
        rows = [
            f"- {handle}（{candidate.get('kind')}）：" +
            "；".join(
                f"{key}={value}"
                for key, value in candidate.items()
                if key not in ("kind", "subject_id") and value
            )
            for handle, candidate in cmap.items()
        ]
        task = load_prompt_template(
            "fragment_kb_task_candidates_v1.md",
            strict=True,
            candidate_rows="\n".join(rows),
            verdicts="|".join(CANDIDATE_VERDICTS),
        )
    return load_prompt_template(
        "kb_repair_v1.md",
        strict=True,
        judgment=load_prompt_template("fragment_kb_judgment_v1.md", strict=True),
        entry_text=entry_text,
        task_section=task,
        ops_contract=load_prompt_template(
            "fragment_knowledge_output_v1.md",
            strict=True,
            reasoning_clause=reasoning_clause(),
        ),
    )


def run_repair_session(
    repo: KnowledgeRepo,
    subject_id: str,
    *,
    client,  # type: ignore[no-untyped-def]
    candidates: list[Mapping[str, Any]] | None = None,
    material: str = "",
    user_prompt: str = "",
    apply: bool = False,
    task_id: str = "kb-repair",
) -> dict[str, Any]:
    """One repair call for one subject. Without ``apply`` the proposals are
    returned for the human to inspect and NOTHING is booked (dry-run
    discipline); with it they go through the normal ``apply_model_proposals``
    path (handle-validated, CAS-checked) and the candidate verdicts land in
    the decision ledger (plan A6)."""

    from ...routing.config import LLMRole
    from .model import CATEGORIES
    from .proposals import apply_model_proposals

    handles = HandleMap()
    cmap = candidate_handle_map(list(candidates or []))
    prompt = render_repair_prompt(
        repo, subject_id, candidates=candidates, material=material,
        user_prompt=user_prompt, handles=handles, candidate_handles=cmap,
    )
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": prompt}],
        max_tokens=REPAIR_MAX_TOKENS,
        task_group="knowledge",
    )
    out: dict[str, Any] = {"subject_id": subject_id, "proposals_text": result.content}
    verdicts = parse_candidate_verdicts(result.content, set(cmap)) if cmap else []
    out["candidate_verdicts"] = verdicts
    if apply:
        report = apply_model_proposals(
            result.content,
            repo=repo,
            task_id=task_id,
            knowledge_read_rev=repo.rev,
            handles=handles,
            # a human opened this session on a named entry, so every category
            # is in scope — including the ones no automatic prompt is wired to
            allow_categories=CATEGORIES,
        )
        out["apply_report"] = report.to_dict()
        if cmap:
            out["candidate_ledger"] = _book_candidate_verdicts(
                repo, cmap, verdicts, result.content, report, task_id
            )
    return out


def _book_candidate_verdicts(
    repo: KnowledgeRepo,
    cmap: Mapping[str, Mapping[str, Any]],
    verdicts: list[dict[str, str]],
    content: str,
    report,  # ProposalReport  # type: ignore[no-untyped-def]
    task_id: str,
) -> list[dict[str, str]]:
    """Ledger半：dismiss → resolved(dismissed)、needs_human → pending_human；
    ``propose`` resolves ONLY when the POST-APPLY scan no longer produces the
    candidate (review 2026-08-28 P1-2: "any op landed" closed candidates whose
    core fix was rejected — e.g. the removal half of a relation migration lost
    to a CAS conflict while the append half landed). The completion assertion
    is the scan itself: resolved(applied) must mean the underlying condition
    is gone. A candidate the scan still finds stays open and resurfaces.

    All ledger writes ride ONE transaction (review P1-3): a crash cannot leave
    the ledger half-written. Cross-step atomicity with the proposal apply is
    deliberately not needed — with scan-based resolution the ledger only caches
    decisions over a re-derivable scan: a crash between apply and booking loses
    nothing (a fixed candidate simply never re-enters the scan; a judged-only
    candidate re-surfaces and is re-judged)."""

    from .candidates import candidate_identity, record_candidate_decision

    remaining = {
        candidate_identity(cand)[0]
        for cand in scan_candidates(repo.store).candidates
    }
    conn = repo.store.conn
    booked: list[dict[str, str]] = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        for row in verdicts:
            candidate = cmap[row["candidate"]]
            key, content_digest = candidate_identity(candidate)
            verdict = row["verdict"]
            # pending_human without a stated evidence gap defeats the whole
            # queue (review 2026-08-28 P2-2): fall back to the reason; a row
            # with NEITHER carries nothing a human can act on — stay open.
            missing = str(row.get("missing", "")).strip() or row["reason"].strip()
            if verdict == "dismiss":
                status, resolution = "resolved", "dismissed"
            elif verdict == "needs_human":
                if not missing:
                    booked.append({"candidate": row["candidate"], "status": "open",
                                   "note": "needs_human 未说明缺什么证据（missing/reason 均空），保持未决"})
                    continue
                status, resolution = "pending_human", ""
            elif not report.rolled_back and key not in remaining:
                status, resolution = "resolved", "applied"
            else:
                booked.append({"candidate": row["candidate"], "status": "open",
                               "note": "apply 后扫描仍产出该候选（修复未落地或不完整），保持未决"})
                continue
            record_candidate_decision(
                repo.store, candidate_key=key, content_digest=content_digest,
                status=status, resolution=resolution, reason=row["reason"],
                task_id=task_id, candidate=candidate,
                missing=missing if verdict == "needs_human" else str(row.get("missing", "")),
            )
            booked.append({"candidate": row["candidate"], "status": status,
                           **({"resolution": resolution} if resolution else {})})
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return booked
