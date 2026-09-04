"""Judgement-call scan (plan §7.3/§11.2, kb-line-grammar plan §8).

What this module produces is CANDIDATES only: lines a human or the repair
session has to decide about. The deterministic half it used to carry moved
to ``phase_b`` when the v3 grammar landed — one converter, not two that can
drift apart:

* ``staging-line`` — anything sitting in a ``staging`` section by construction
* ``unnamed-term`` — a term with no 中文定名 (Phase B leaves the column empty
  when it converts a v1 relation, and the line may not even belong there)
* ``episodic-desc`` — a desc that reads like version narrative
* ``duplicate-term`` — same surface twice under one subject
* ``chapter-alias`` — a subject alias that is also a term surface
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .matching import scan_normalize
from .store import KnowledgeStore

_EPISODIC_MARKERS = ("Ver.", "版本", "新皮肤", "活动中", "近期", "最近")


@dataclass
class CandidateScan:
    """Candidates for a human (or the repair session) to act on through the
    normal edit/proposal surfaces. Nothing here is applied automatically."""

    rev: int
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rev": self.rev, "candidates": self.candidates}


def _fold(text: str) -> str:
    return scan_normalize(text or "").replace(" ", "")


def scan_candidates(store: KnowledgeStore, rev: int | None = None) -> CandidateScan:
    at = store.current_rev() if rev is None else rev
    plan = CandidateScan(rev=at)

    items_by_node: dict[str, list] = {}
    for item in store.all_items(at):
        items_by_node.setdefault(item.local_id, []).append(item)

    def _has_item(local_id: str, field_name: str, value: str) -> bool:
        norm = _fold(value)
        return any(
            i.field == field_name and _fold(i.value) == norm
            for i in items_by_node.get(local_id, [])
        )

    subjects = {s.local_id: s for s in store.subjects(at)}
    term_surfaces = {
        _fold(t.payload.get("surface", "")) for t in store.nodes_of_kind("term", at)
    }
    # Every candidate carries its owning subject_id explicitly (plan A6 /
    # review P1-2): repair routing must never guess the owner — a candidate
    # without one would be silently dropped. Shared nodes keep the first
    # parent, same as the old parents[0] walk.
    owner_of: dict[str, str] = {}
    for subject_id in subjects:
        for membership in store.children(subject_id, at):
            owner_of.setdefault(membership.child_id, subject_id)

    from .presets import preset_for_category

    for subject in subjects.values():
        preset = preset_for_category(str(subject.payload.get("category", "")) or "common")
        # staging sections are a transfer point, not a warehouse: every line
        # sitting in one is a candidate by construction (preset `staging`)
        for membership in store.children(subject.local_id, at):
            spec = preset.section(membership.section)
            if spec is None or not spec.staging:
                continue
            child = store.node(membership.child_id, at)
            if child is None:
                continue
            from .render import format_line

            plan.candidates.append(
                {"kind": "staging-line", "node": child.local_id,
                 "subject": subject.payload.get("surface", ""),
                 "subject_id": subject.local_id,
                 "section": membership.section,
                 "line": format_line(child),
                 "hint": f"「{membership.section}」是暂存区：把这行归位到合适的小节"
                         "（需要中文定名/别名就补上，术语节要写成四列术语行），"
                         "或按该节的收录纪律删除"}
            )
        # a term Phase B produced from a v1 relation: its 中文定名 column is
        # empty by construction, and the line may not belong in 人际关系 at all
        # (channel coinage was admitted there under the old kind rules)
        for membership in store.children(subject.local_id, at):
            child = store.node(membership.child_id, at)
            if child is None or child.kind != "term":
                continue
            if str(child.payload.get("zh") or "").strip():
                continue
            plan.candidates.append(
                {"kind": "unnamed-term", "node": child.local_id,
                 "subject": subject.payload.get("surface", ""),
                 "subject_id": subject.local_id,
                 "section": membership.section,
                 "surface": child.payload.get("surface", ""),
                 "desc": str(child.payload.get("desc", ""))[:160],
                 "hint": "缺中文定名：补上官方或社区公认译名（查不到就留空并在描述里注明「音译，未定名」）；"
                         "若这其实是频道自造词而非人/组织，整行迁去「频道用语」"}
            )
        # chapter-alias candidates: a subject alias that is also a term surface
        for item in items_by_node.get(subject.local_id, []):
            if item.field == "aliases" and _fold(item.value) in term_surfaces:
                plan.candidates.append(
                    {"kind": "chapter-alias", "subject": subject.payload.get("surface", ""),
                     "subject_id": subject.local_id,
                     "item_id": item.item_id, "value": item.value,
                     "hint": "版本篇章名挂在 subject 别名上，且已有同名独立 term——建议移除该别名"}
                )

    for term in store.nodes_of_kind("term", at):
        payload = term.payload
        label = str(payload.get("surface", ""))
        stripped = str(payload.get("desc", ""))
        if any(marker in stripped for marker in _EPISODIC_MARKERS):
            plan.candidates.append(
                {"kind": "episodic-desc", "term": label, "node": term.local_id,
                 "subject_id": owner_of.get(term.local_id, ""),
                 "desc": stripped[:120],
                 "hint": "desc 疑含版本性叙事——身份稳定的一句话之外的内容会随版本过期，建议删除"}
            )

    # duplicate-term candidates: two live terms with the same folded surface
    # under one subject render as conflicting lines (the real store carries
    # e.g. ロスカリファ twice with two contradictory zh spellings). Merging is
    # a judgment call — candidates only, the repair session decides.
    for subject in subjects.values():
        by_surface: dict[str, dict[str, Any]] = {}
        for membership in store.children(subject.local_id, at):
            child = store.node(membership.child_id, at)
            if child is not None and child.kind == "term":
                fold = _fold(str(child.payload.get("surface", "")))
                if fold:
                    by_surface.setdefault(fold, {})[child.local_id] = child
        for dupes in by_surface.values():
            if len(dupes) > 1:
                first = next(iter(dupes.values()))
                plan.candidates.append(
                    {"kind": "duplicate-term", "node": first.local_id,
                     "subject": subject.payload.get("surface", ""),
                     "subject_id": subject.local_id,  # explicit routing: never guessed from parents[0]
                     "surface": first.payload.get("surface", ""),
                     "nodes": sorted(dupes),
                     "hint": "同一 subject 下同名 term 多条，译名/desc 可能互相矛盾——建议合并或改名"}
                )

    # The legacy-kind half of this pass moved to `phase_b` (plan §8): the
    # deterministic sep normalization is GONE for good — it rewrote content
    # whenever the v1 fact regex had split a prose line on a colon inside a
    # title (《崩坏：星穹铁道》 became 《崩坏: 星穹铁道》). Phase B puts such a
    # line back together instead of normalizing the bogus separator, and the
    # grab-bag / relation candidates it used to raise are handled there too.
    return plan
