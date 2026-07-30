"""Time-based alignment between a machine SRT and a user-refined SRT.

The refined SRT may contain non-audio annotations or intentionally skip parts
of the transcript, so knowledge updates must only rely on clearly matched
pairs. This module classifies segments mechanically by time overlap:

- ``matched``: machine segment(s) clearly corresponding to one refined segment.
- ``refined_only``: refined segments with no machine counterpart (annotations,
  inserts) — ignorable as correction evidence.
- ``machine_only``: machine segments the user skipped — equally ignorable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .model import SrtSegment, format_srt_timestamp


@dataclass(frozen=True)
class AlignedPair:
    machine_segments: Tuple[SrtSegment, ...]
    refined_segment: SrtSegment

    @property
    def machine_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.machine_segments)

    @property
    def texts_differ(self) -> bool:
        return _normalize(self.machine_text) != _normalize(self.refined_segment.text)


@dataclass(frozen=True)
class AlignmentResult:
    matched: List[AlignedPair]
    refined_only: List[SrtSegment]
    machine_only: List[SrtSegment]


def _normalize(text: str) -> str:
    return "".join((text or "").split())


def _overlap_seconds(a: SrtSegment, b: SrtSegment) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _duration(segment: SrtSegment) -> float:
    return max(1e-6, segment.end - segment.start)


def align_srt(
    machine: Sequence[SrtSegment],
    refined: Sequence[SrtSegment],
    *,
    # 0.55 (not 0.5): when the user merges two lines into one, each half covers
    # at most ~half of the merged line (IoU <= ~0.5) and must fall through to
    # the pass-2 grouping instead of claiming a 1:1 match.
    iou_threshold: float = 0.55,
    contain_threshold: float = 0.7,
    group_member_threshold: float = 0.6,
    group_coverage_threshold: float = 0.6,
) -> AlignmentResult:
    """Deterministic two-pass alignment.

    Pass 1 takes one-to-one pairs greedily by overlap seconds (IoU or
    containment above threshold). Pass 2 groups leftover machine segments that
    mostly fall inside one leftover refined segment (the user merged lines).
    """
    matched_machine: set[int] = set()
    matched_refined: set[int] = set()
    pairs: List[Tuple[float, int, int]] = []
    for m_idx, m_seg in enumerate(machine):
        for r_idx, r_seg in enumerate(refined):
            overlap = _overlap_seconds(m_seg, r_seg)
            if overlap <= 0:
                continue
            union = _duration(m_seg) + _duration(r_seg) - overlap
            iou = overlap / union if union > 0 else 0.0
            # Coverage of the LARGER side: a short machine line inside a long
            # refined line must not claim it 1:1 — that case belongs to the
            # pass-2 grouping (the user merged lines).
            coverage = overlap / max(_duration(m_seg), _duration(r_seg))
            if iou >= iou_threshold or coverage >= contain_threshold:
                pairs.append((overlap, m_idx, r_idx))
    one_to_one: dict[int, int] = {}
    for overlap, m_idx, r_idx in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if m_idx in matched_machine or r_idx in matched_refined:
            continue
        matched_machine.add(m_idx)
        matched_refined.add(r_idx)
        one_to_one[r_idx] = m_idx

    grouped: dict[int, List[int]] = {}
    for r_idx, r_seg in enumerate(refined):
        if r_idx in matched_refined:
            continue
        members = [
            m_idx
            for m_idx, m_seg in enumerate(machine)
            if m_idx not in matched_machine
            and _overlap_seconds(m_seg, r_seg) >= group_member_threshold * _duration(m_seg)
        ]
        if not members:
            continue
        coverage = sum(_overlap_seconds(machine[m_idx], r_seg) for m_idx in members)
        if coverage >= group_coverage_threshold * _duration(r_seg):
            grouped[r_idx] = members
            matched_refined.add(r_idx)
            matched_machine.update(members)

    matched: List[AlignedPair] = []
    for r_idx, r_seg in enumerate(refined):
        if r_idx in one_to_one:
            matched.append(AlignedPair((machine[one_to_one[r_idx]],), r_seg))
        elif r_idx in grouped:
            members = tuple(machine[m_idx] for m_idx in sorted(grouped[r_idx]))
            matched.append(AlignedPair(members, r_seg))
    matched.sort(key=lambda pair: pair.refined_segment.start)

    refined_only = [seg for idx, seg in enumerate(refined) if idx not in matched_refined]
    machine_only = [seg for idx, seg in enumerate(machine) if idx not in matched_machine]
    return AlignmentResult(matched=matched, refined_only=refined_only, machine_only=machine_only)


def _time_label(start: float, end: float) -> str:
    return f"[{format_srt_timestamp(start)} - {format_srt_timestamp(end)}]"


def render_alignment_diff(
    result: AlignmentResult,
) -> str:
    """Chinese report for prompt injection: only clearly matched pairs count as
    evidence; the other two sections are listed but flagged ignorable."""
    differing = [pair for pair in result.matched if pair.texts_differ]
    lines: List[str] = [
        "# 机器字幕 vs 精修字幕 时间对齐报告",
        f"- 明确对应: {len(result.matched)} 组（其中 {len(differing)} 组文本有差异）",
        f"- 仅精修侧（注释/新增，默认忽略）: {len(result.refined_only)} 条",
        f"- 仅机器侧（用户跳过，默认忽略）: {len(result.machine_only)} 条",
        "",
        "## 明确对应且文本有差异（知识/错误提炼的唯一依据）",
    ]
    for pair in differing:
        refined = pair.refined_segment
        lines.append(
            f"- {_time_label(refined.start, refined.end)} "
            f"机器: 「{pair.machine_text}」 ⇔ 精修: 「{refined.text.strip()}」"
        )
    lines.extend(["", "## 仅精修侧（默认忽略）"])
    for segment in result.refined_only:
        lines.append(f"- {_time_label(segment.start, segment.end)} 「{segment.text.strip()}」")
    lines.extend(["", "## 仅机器侧（默认忽略）"])
    for segment in result.machine_only:
        lines.append(f"- {_time_label(segment.start, segment.end)} 「{segment.text.strip()}」")
    return "\n".join(lines) + "\n"
