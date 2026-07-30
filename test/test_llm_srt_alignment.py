from __future__ import annotations

from asr_playground.subtitles.alignment import align_srt, render_alignment_diff
from asr_playground.subtitles.model import SrtSegment


def _seg(index: int, start: float, end: float, text: str) -> SrtSegment:
    return SrtSegment(index=index, start=start, end=end, text=text)


def test_exact_one_to_one_matches() -> None:
    machine = [_seg(1, 0.0, 2.0, "你好"), _seg(2, 3.0, 5.0, "再见")]
    refined = [_seg(1, 0.0, 2.0, "你好呀"), _seg(2, 3.0, 5.0, "再见")]

    result = align_srt(machine, refined)

    assert len(result.matched) == 2
    assert not result.refined_only and not result.machine_only
    assert result.matched[0].texts_differ
    assert not result.matched[1].texts_differ


def test_shifted_timing_still_matches_by_iou() -> None:
    machine = [_seg(1, 10.0, 14.0, "台词")]
    refined = [_seg(1, 10.8, 14.5, "精修台词")]

    result = align_srt(machine, refined)

    assert len(result.matched) == 1
    assert result.matched[0].machine_segments[0].text == "台词"


def test_refined_annotation_and_machine_skip_are_flagged_ignorable() -> None:
    machine = [_seg(1, 0.0, 2.0, "你好"), _seg(2, 30.0, 32.0, "用户跳过的内容")]
    refined = [_seg(1, 0.0, 2.0, "你好"), _seg(2, 10.0, 12.0, "（屏幕注释）")]

    result = align_srt(machine, refined)

    assert [seg.text for seg in result.refined_only] == ["（屏幕注释）"]
    assert [seg.text for seg in result.machine_only] == ["用户跳过的内容"]


def test_user_merged_lines_match_as_group() -> None:
    machine = [_seg(1, 0.0, 1.0, "前半"), _seg(2, 1.2, 2.4, "后半")]
    refined = [_seg(1, 0.0, 2.4, "前半后半合并")]

    result = align_srt(machine, refined)

    assert len(result.matched) == 1
    pair = result.matched[0]
    assert [seg.text for seg in pair.machine_segments] == ["前半", "后半"]
    assert pair.machine_text == "前半 后半"


def test_alignment_is_deterministic() -> None:
    machine = [_seg(i, i * 2.0, i * 2.0 + 1.5, f"m{i}") for i in range(10)]
    refined = [_seg(i, i * 2.0 + 0.2, i * 2.0 + 1.7, f"r{i}") for i in range(10)]

    first = align_srt(machine, refined)
    second = align_srt(machine, refined)

    assert [p.refined_segment.text for p in first.matched] == [
        p.refined_segment.text for p in second.matched
    ]
    assert len(first.matched) == 10


def test_render_alignment_diff_sections() -> None:
    machine = [_seg(1, 0.0, 2.0, "你好"), _seg(2, 30.0, 32.0, "跳过")]
    refined = [_seg(1, 0.0, 2.0, "你好呀"), _seg(2, 10.0, 12.0, "（注释）")]

    text = render_alignment_diff(align_srt(machine, refined))

    assert "明确对应: 1 组（其中 1 组文本有差异）" in text
    assert "仅精修侧（注释/新增，默认忽略）: 1 条" in text
    assert "仅机器侧（用户跳过，默认忽略）: 1 条" in text
    assert "「你好」 ⇔ 精修: 「你好呀」" in text
