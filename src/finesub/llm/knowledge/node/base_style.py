"""Overlaying the factory default style onto the styles a user already has
(owner 2026-11-03; docs/llm_design_notes.md「出厂默认风格的决策记录」).

The tendencies the correction and proofread rules used to carry moved into
the factory default style, which is injected only when no user style is in
effect. A style written before that has none of them, so a one-time LLM task
adds to it whatever of the factory default it does not already say or
contradict -- as a new revision of the same entry, which `knowledge revert`
can undo.

The task gets its own prompt (`kb_base_style_v1.md`), not the repair one: the
repair session's recording standard, writing rules and full op grammar all
argue against copying a generic line into an entry. What it may write is
screened again before translation (`admit_proposals`): only `append_lines`,
only into this style, only into the sections the factory default seeds.

Which styles still wait for it is a `meta` key (`PENDING_META_KEY`), not a
schema version: a worktree opening the main checkout's store read-only would
otherwise migrate it past what `dev` code can open. The first open by this
code records the live styles (`seed_pending`); an applied migration crosses
its style off, and a correction run naming a style still on the list warns.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from typing import Any, Mapping, Sequence

from finesub import jsonl

from .presets import preset_for_category
from .render import HandleMap
from .repo import KnowledgeRepo
from .store import KnowledgeStore

PENDING_META_KEY = "factory_style_pending"
TASK_ID = "kb-base-style"
COMMAND = "python -m finesub.llm.knowledge adopt-base-style"
#: what a `<skipped>` row may give as its reason -- rendered into the prompt
#: from here so the parser and the prompt cannot disagree
SKIP_REASONS = ("同标记", "已说过", "相反", "部分相反", "拿不准", "约定没加", "放不下")

_STYLE = "style"
_LABEL_RE = re.compile(r"^\[([^\]]+)\]")
_SKIPPED_RE = re.compile(r"<skipped\b[^>]*>(?P<body>.*?)</skipped>", re.IGNORECASE | re.DOTALL)


# ---- the pending list ------------------------------------------------------------------


def _live_styles(store: KnowledgeStore) -> list[str]:
    return [s.local_id for s in store.subjects() if s.payload.get("category") == _STYLE]


def seed_pending(store: KnowledgeStore) -> None:
    """Record the live styles as waiting for the factory default, once per store.

    Check and write ride one IMMEDIATE transaction, so two processes opening a
    store for the first time at once write the key once. A style created
    after this point is not on the list: `knowledge new style` seeds the
    factory default itself."""

    if store.meta(PENDING_META_KEY) is not None:
        return
    conn = store.conn
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        # Another process held the write lock past the busy timeout: record
        # on a later open rather than fail this one over a reminder list.
        return
    try:
        if store.meta(PENDING_META_KEY) is None:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                (PENDING_META_KEY, json.dumps(_live_styles(store))),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def pending_styles(store: KnowledgeStore) -> list[str]:
    """Local ids of the styles still waiting, in the order they were recorded."""

    raw = store.meta(PENDING_META_KEY)
    return [str(i) for i in json.loads(raw)] if raw else []


def clear_pending(store: KnowledgeStore, local_id: str) -> None:
    conn = store.conn
    conn.execute("BEGIN IMMEDIATE")
    try:
        ids = pending_styles(store)
        if local_id in ids:
            conn.execute(
                "UPDATE meta SET value=? WHERE key=?",
                (json.dumps([i for i in ids if i != local_id]), PENDING_META_KEY),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


_warned: set[tuple[str, str]] = set()
_warned_lock = threading.Lock()


def warn_if_pending(repo: KnowledgeRepo, subject_id: str, key: str) -> None:
    """One warning per style and process: a run reads its style once per window."""

    if subject_id not in pending_styles(repo.store):
        return
    with _warned_lock:
        if (str(repo.root), subject_id) in _warned:
            return
        _warned.add((str(repo.root), subject_id))
    from finesub.reporting import current_reporter

    current_reporter().warning(
        "knowledge-style-not-migrated",
        f"风格「{key}」写成时还没有出厂默认风格，还没迁移",
        impact="纠错、校对的通用规则里原有的口味要求已挪进出厂默认风格，而用了这套风格时出厂默认不注入，这些要求这次没有",
        action=f"{COMMAND} --subject style/{key}（不带旗标只渲染 prompt，--execute 看提案，--apply 落库）",
    )


# ---- the session -----------------------------------------------------------------------


def _section_limits(repo: KnowledgeRepo, subject_id: str) -> str:
    """Lines left per seed section, from the preset. The line length cap is not
    stated: the factory default's lines fit it, and a part of one is shorter."""

    from ..style import FACTORY_SEED_SECTIONS

    preset = preset_for_category(_STYLE)
    children = repo.store.children(subject_id)
    parts: list[str] = []
    for section in FACTORY_SEED_SECTIONS:
        spec = preset.section(section)
        left = "不限" if spec is None or spec.max_lines is None else \
            f"{max(0, spec.max_lines - sum(1 for m in children if m.section == section))} 行"
        parts.append(f"「{section}」{left}")
    return "、".join(parts)


def render_base_style_prompt(repo: KnowledgeRepo, subject_id: str, handles: HandleMap) -> str:
    from ...prompt_compose import load_prompt_template
    from ..style import factory_style_text
    from .render import render_subject

    # The partial preview: the full one's section comments carry the update
    # task's recording standard ("lines true of any style are not recorded"),
    # which is the opposite of this task.
    entry_text = render_subject(
        repo.store, subject_id, mode="prompt", handles=handles, alias_row_handle=False,
    )
    subject = repo.store.node(subject_id)
    assert subject is not None, subject_id
    return load_prompt_template(
        "kb_base_style_v1.md",
        strict=True,
        entry_text=entry_text,
        factory_text=factory_style_text(),
        entry_handle=handles.node_handle(subject),
        limits=_section_limits(repo, subject_id),
        skip_reasons="|".join(SKIP_REASONS),
    ).strip()


def admit_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    repo: KnowledgeRepo,
    subject_id: str,
    handles: HandleMap,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split parsed proposals into the lines this task may write and refusals.

    Screened here, before translation, not on the engine ops: `translate`
    adds an `updated_date` update of its own to every entry it touches. Each
    content line is screened on its own and admitted as a proposal of its own:

    * only `append_lines`, into this style (by handle, or by key pinned to the
      style namespace, which a bare key would otherwise miss), into a seed section;
    * its `[标记]` is one the factory default has in that section -- the prompt
      keeps the labels, so another one is a line the model made up;
    * an example only beside a convention of the same label admitted in this
      batch: the factory default's examples come with their convention, and one
      without it would make the engine roll the whole batch back as an orphan.
    """

    from ..style import FACTORY_SEED_SECTIONS, factory_style_lines

    factory_labels = {
        section: {m.group(1) for line in lines if (m := _LABEL_RE.match(line))}
        for section, lines in factory_style_lines().items()
    }
    admitted: list[dict[str, Any]] = []
    refused: list[str] = []

    def _refuse(problem: str, what: Mapping[str, Any]) -> None:
        refused.append(f"{problem}：{json.dumps(dict(what), ensure_ascii=False)[:200]}")

    for proposal in proposals:
        op = str(proposal.get("op", ""))
        entry = str(proposal.get("entry", ""))
        category = proposal.get("category")
        section = str(proposal.get("section", "")).strip()
        if op != "append_lines":
            problem = "只收 append_lines" if op != "__invalid__" else "不是 JSON"
        elif section not in FACTORY_SEED_SECTIONS:
            problem = "只收「" + "」「".join(FACTORY_SEED_SECTIONS) + "」三节"
        elif category not in (None, "", _STYLE):
            problem = "只收这套风格"
        elif entry.startswith("@"):
            bound = handles.nodes.get(entry)
            problem = "" if bound is not None and bound[0] == subject_id else "只收这套风格"
        else:
            resolved = repo.resolve(entry, category=_STYLE)
            problem = "" if resolved is not None and resolved.subject_id == subject_id else "只收这套风格"
        if problem:
            _refuse(problem, proposal)
            continue
        for line in (row.strip() for row in str(proposal.get("content", "")).splitlines()):
            if not line:
                continue
            match = _LABEL_RE.match(line)
            if match is None or match.group(1) not in factory_labels.get(section, set()):
                _refuse(f"标记不是出厂默认「{section}」里的", {**proposal, "content": line})
                continue
            admitted.append({**proposal, "category": _STYLE, "section": section, "content": line})
    conventions = {_LABEL_RE.match(p["content"]).group(1) for p in admitted if p["section"] == "约定"}  # type: ignore[union-attr]
    kept: list[dict[str, Any]] = []
    for proposal in admitted:
        if proposal["section"] != "约定" and _LABEL_RE.match(proposal["content"]).group(1) not in conventions:  # type: ignore[union-attr]
            _refuse("例子的约定这次没加", proposal)
        else:
            kept.append(proposal)
    return kept, refused


def reply_complete(text: str) -> bool:
    """Both answer blocks are there and closed: a reply that lacks one is a
    broken reply, not the model's judgement that nothing should be added."""

    from .proposals import _PROPOSALS_RE, strip_reasoning

    body = strip_reasoning(text)
    return bool(_PROPOSALS_RE.search(body) and _SKIPPED_RE.search(body))


def parse_skipped(text: str) -> list[dict[str, str]]:
    """Rows of the `<skipped>` block; a row that is not a JSON object is kept
    as its raw text, since this block is only ever shown to a human."""

    from .proposals import strip_reasoning

    match = _SKIPPED_RE.search(strip_reasoning(text))
    if not match:
        return []
    rows: list[dict[str, str]] = []
    for line in jsonl.lines(match.group("body")):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            rows.append({"raw": line})
            continue
        row = {key: str(data.get(key) or "").strip() for key in ("label", "section", "why", "line")}
        # The report prints the label in its own brackets (`maintain.py`); the
        # prompt does not say whether to write them (cold read round 4, P501).
        row["label"] = row["label"].strip("[] ")
        rows.append(row)
    return rows


def run_base_style_session(
    repo: KnowledgeRepo,
    subject_id: str,
    *,
    client,  # type: ignore[no-untyped-def]
    apply: bool = False,
) -> dict[str, Any]:
    """One call for one style. Without ``apply`` nothing is written; with it the
    admitted proposals go through `apply_proposals`, and the style leaves the
    pending list when the answer was whole and landed clean -- also when
    nothing was added. ``kept_pending`` says why it stays on otherwise."""

    from ...client import OutputCutOff, reply_cut_off
    from ...routing.config import SESSION_OUTPUT_MAX_TOKENS, LLMRole
    from .proposals import apply_proposals, parse_model_proposals

    handles = HandleMap()
    read_rev = repo.rev
    prompt = render_base_style_prompt(repo, subject_id, handles)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": prompt}],
        output_reserve=SESSION_OUTPUT_MAX_TOKENS,
        task_group="knowledge",
    )
    if reply_cut_off(result):
        subject = repo.store.node(subject_id)
        surface = subject.payload.get("surface", subject_id) if subject else subject_id
        raise OutputCutOff(f"{TASK_ID} {surface}", result.requested_output_tokens)
    complete = reply_complete(result.content)
    # A reply without its blocks is not screened: the parser would fall back
    # to reading every line of it as a proposal, and list each as refused.
    admitted, refused = admit_proposals(
        parse_model_proposals(result.content), repo=repo, subject_id=subject_id, handles=handles,
    ) if complete else ([], [])
    out: dict[str, Any] = {
        "subject_id": subject_id,
        "proposals": admitted,
        "refused": refused,
        "skipped": parse_skipped(result.content),
    }
    if apply:
        before = {m.child_id for m in repo.store.children(subject_id)}
        report = apply_proposals(
            admitted, repo=repo, allow_categories=(_STYLE,), task_id=TASK_ID,
            knowledge_read_rev=read_rev, handles=handles,
        )
        out["apply_report"] = report.to_dict()
        # What landed, read back from the store: a row the engine skipped
        # (a label already in the section) is in a proposal but not here.
        out["added"] = _added_lines(repo, subject_id, before)
        # Off the list only on a whole, clean answer that landed: also when it
        # added nothing, never when the reply was broken or something was refused.
        out["kept_pending"] = (
            "整批回滚" if report.rolled_back
            else "回答缺 <knowledge_proposals> 或 <skipped> 块" if not complete
            else "有提案被拒收" if refused
            else ""
        )
        if not out["kept_pending"]:
            clear_pending(repo.store, subject_id)
    return out


def _added_lines(repo: KnowledgeRepo, subject_id: str, before: set[str]) -> list[tuple[str, str]]:
    from .render import format_line

    out: list[tuple[str, str]] = []
    for membership in repo.store.children(subject_id):
        if membership.child_id in before:
            continue
        node = repo.store.node(membership.child_id)
        if node is not None:
            out.append((membership.section, format_line(node)))
    return out
