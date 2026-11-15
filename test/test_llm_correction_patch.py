"""The correction window's patch repair round (`correction_patch`)."""

from __future__ import annotations

import pytest

from finesub.llm.chunking import Seam, SubtitleSegment, SubtitleWindow
from finesub.llm.correction_patch import apply_patch, patch_instruction, repair_base, rows_to_fix
from finesub.llm.output_protocol import (
    correction_output_header,
    validate_correction_window_output,
)
from finesub.llm.prompt_compose import compose_repair_turns
from finesub.llm.prompt_variants import resolve_variant

CAPABLE_C = resolve_variant("capableC")
HEADER = correction_output_header(CAPABLE_C)  # six columns


def _window(count: int = 6, seams: dict[int, Seam] | None = None) -> SubtitleWindow:
    # Canonical ids that are not the local 1..N, so a mix-up between the two shows.
    segments = [
        SubtitleSegment(f"s{100 + index}", index * 2.0, index * 2.0 + 1.5, f"原文{index}",
                        seam=(seams or {}).get(index, Seam.NONE if seams else None))
        for index in range(1, count + 1)
    ]
    return SubtitleWindow("0001", segments, [], "test", None, length_scale=1.0)  # type: ignore[arg-type]


def _sub(position: str, translation: str | None = None) -> str:
    text = translation or f"译文{position.replace(',', '')}"
    return f"sub|{position}|源{position}|{text}|high|"


def _answer(rows: list[str], *, header: str = HEADER, reasoning: str = "全局判断") -> str:
    body = "\n".join([header, *rows])
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<translated>\n{body}\n</translated>\n"


def _patch(rows: list[str], *, header: str = HEADER) -> str:
    return "<patch>\n" + "\n".join([header, *rows]) + "\n</patch>"


def _base(text: str, window: SubtitleWindow, variant=CAPABLE_C):
    result = validate_correction_window_output(text, window, variant=variant)
    assert not result.ok
    return result, repair_base(text, result, window, variant)


def test_each_row_carries_its_own_errors() -> None:
    window = _window(4)
    text = _answer(["# 2、3 同一句", _sub("1"), "sub|2|oops", _sub("3") + "<void>", _sub("4")])
    result = validate_correction_window_output(text, window, variant=CAPABLE_C)

    kinds = [(verdict.kind, verdict.ok) for verdict in result.rows]
    assert kinds == [("comment", True), ("sub", True), ("sub", False), ("void", True), ("sub", True)]
    assert result.rows[2].errors and all("Row 3" in error for error in result.rows[2].errors)
    # Every row error is one of the rows'; the rest is about the window as a whole.
    row_errors = [error for verdict in result.rows for error in verdict.errors]
    assert [error for error in result.errors if error not in row_errors] == [
        error for error in result.errors if "missing source" in error
    ]


def test_a_patch_fills_a_missing_source_and_keeps_every_other_row() -> None:
    window = _window()
    text = _answer([_sub("1"), _sub("2"), _sub("4"), _sub("5"), _sub("6")])
    _result, base = _base(text, window)

    assert base is not None and base.pending == ("3",)
    merged = apply_patch(base, _patch([_sub("3")]))

    assert merged.mode == "patch" and merged.patch_units == 1 and merged.replaced_units == 0
    assert validate_correction_window_output(merged.text, window, variant=CAPABLE_C).ok
    rows = merged.text.split("<translated>\n", 1)[1].split("\n</translated>", 1)[0].splitlines()
    assert rows == [HEADER, _sub("1"), _sub("2"), _sub("3"), _sub("4"), _sub("5"), _sub("6")]
    assert "<reasoning>\n全局判断\n</reasoning>" in merged.text


def test_a_patch_touching_a_merged_row_takes_the_whole_row_out() -> None:
    window = _window()
    # 3 is written twice: the second row fails, the merged first one passes.
    text = _answer([_sub("1"), _sub("2,3"), _sub("3"), _sub("4"), _sub("5"), _sub("6")])
    _result, base = _base(text, window)
    assert base is not None and base.pending == ("3",)

    only_three = apply_patch(base, _patch([_sub("3")]))
    assert only_three.mode == "patch" and only_three.replaced_units == 1
    result = validate_correction_window_output(only_three.text, window, variant=CAPABLE_C)
    assert not result.ok and any("missing source id(s): 2." in error for error in result.errors)  # model-local ids

    both = apply_patch(base, _patch([_sub("2"), _sub("3")]))
    assert validate_correction_window_output(both.text, window, variant=CAPABLE_C).ok


def test_comment_lines_go_with_their_row() -> None:
    window = _window(4)
    text = _answer(["# 1 独立", _sub("1"), "# 2 的说明", "sub|2|bad", "# 3、4 合并", _sub("3,4")])
    _result, base = _base(text, window)
    assert base is not None and base.pending == ("2",)

    merged = apply_patch(base, _patch(["# 2 改好了", _sub("2")]))
    body = merged.text.split("<translated>\n", 1)[1]
    assert "# 2 的说明" not in body  # the failed row's comment left with it
    assert body.index("# 1 独立") < body.index(_sub("1")) < body.index("# 2 改好了") < body.index(_sub("2"))
    assert body.index(_sub("2")) < body.index("# 3、4 合并") < body.index(_sub("3,4"))
    assert validate_correction_window_output(merged.text, window, variant=CAPABLE_C).ok


def test_a_voided_patch_row_takes_nothing_out() -> None:
    window = _window(4)
    _result, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)
    merged = apply_patch(base, _patch([_sub("2,3") + "<void>", _sub("3")]))

    assert merged.replaced_units == 0
    assert validate_correction_window_output(merged.text, window, variant=CAPABLE_C).ok


def test_discard_rows_patch_like_any_row() -> None:
    window = _window(4)
    _result, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)
    merged = apply_patch(base, _patch(["discard|3"]))
    assert validate_correction_window_output(merged.text, window, variant=CAPABLE_C).ok


def test_a_patch_that_cannot_be_used_leaves_the_base_alone() -> None:
    window = _window(4)
    _result, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)

    for reply in (
        _patch([_sub("3")]) + "\n" + _answer([_sub("1")]),  # both
        "我已经改好了",  # neither
        _patch([_sub("0")]),  # read-only context id
        _patch([_sub("9")]),  # past the window
        _patch(["sub||源|译|high|"]),  # no ids
        _patch([]),  # empty
    ):
        merged = apply_patch(base, reply)
        assert merged.mode == "invalid" and merged.errors and merged.text == "", reply


def test_a_whole_rewrite_is_still_taken_as_it_came() -> None:
    window = _window(4)
    _result, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)
    rewrite = _answer([_sub("1"), _sub("2"), _sub("3"), _sub("4")])
    merged = apply_patch(base, rewrite)
    assert merged.mode == "full" and merged.text == rewrite


def test_blocks_the_patch_reply_writes_replace_the_base_ones() -> None:
    window = _window(4)
    _result, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)
    merged = apply_patch(base, "<reasoning>\n补上 3\n</reasoning>\n" + _patch([_sub("3")]) + "\n<next_advice>\n下一窗\n</next_advice>")
    assert "补上 3" in merged.text and "全局判断" not in merged.text
    assert merged.text.rstrip().endswith("<next_advice>\n下一窗\n</next_advice>")


def test_a_start_column_variant_patches_in_its_own_layout() -> None:
    basic_a = resolve_variant("basicA")
    window = _window(4)

    header = correction_output_header(basic_a)

    def row(position: str) -> str:
        return f"sub|{position}|0.0|源{position}|译文{position}|high|"

    text = _answer([row("1"), row("2"), row("4")], header=header)
    _result, base = _base(text, window, basic_a)
    assert base is not None and base.header == header
    merged = apply_patch(base, _patch([row("3")], header=header))
    assert validate_correction_window_output(merged.text, window, variant=basic_a).ok


def test_answers_that_go_back_for_a_rewrite() -> None:
    window = _window(6)
    # Most of the window failed: a systematic format error.
    _result, base = _base(_answer([_sub("1"), _sub("2"), "sub|3|x", "sub|4|x", "sub|5|x", "sub|6|x"]), window)
    assert base is None
    # An error no row owns (the header).
    text = "<translated>\n" + "\n".join([_sub("1"), _sub("2"), _sub("4"), _sub("5"), _sub("6")]) + "\n</translated>"
    _result, base = _base(text, window)
    assert base is None
    # No block at all.
    result = validate_correction_window_output("没有输出", window, variant=CAPABLE_C)
    assert repair_base("没有输出", result, window, CAPABLE_C) is None
    # A variant that also wants `<singles>`.
    basic_a = resolve_variant("basicA")
    text = _answer([_sub("1"), _sub("2"), _sub("4"), _sub("5"), _sub("6")], header=correction_output_header(basic_a))
    result = validate_correction_window_output(text, window, variant=basic_a)
    assert repair_base(text, result, window, basic_a) is None


def test_more_than_twelve_missing_sources_are_all_pending() -> None:
    """The coverage error lists twelve; the pending list is counted from coverage, not read off it."""

    window = _window(40)
    rows = [_sub(str(index)) for index in range(1, 41) if index % 3]
    _result, base = _base(_answer(rows), window)
    assert base is not None and base.pending == tuple(str(index) for index in range(3, 41, 3))


def test_a_merge_across_a_turn_rides_along_in_local_ids() -> None:
    window = _window(5, seams={3: Seam.CHANGE})
    _result, base = _base(_answer([_sub("1"), _sub("2,3"), _sub("5")]), window)
    assert base is not None and base.pending == ("2", "3", "4")
    # The row across the turn is shown; 4 is missing and has none.
    assert rows_to_fix(base) == _sub("2,3")


def test_rows_to_fix_shows_every_row_holding_a_pending_source() -> None:
    # 3 is pending: the merged row that holds it and the failing row of its own both stand to be replaced.
    _result, base = _base(_answer([_sub("1"), _sub("2,3"), _sub("3"), _sub("4"), _sub("5"), _sub("6")]), _window())
    assert base is not None and base.pending == ("3",)
    assert rows_to_fix(base).splitlines() == [_sub("2,3"), _sub("3")]


def test_the_patch_instruction_names_the_sources_and_the_header() -> None:
    window = _window(4)
    _result, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)
    text = patch_instruction(base, ["Translated missing source id(s): 3.", "  "])

    assert "- Translated missing source id(s): 3." in text
    assert "要改的源序号：3。" in text
    assert f"`{HEADER}`" in text
    assert "<patch>" in text and "$" not in text


def test_repair_turns_carry_a_stage_instruction_verbatim() -> None:
    turns = compose_repair_turns("prev", ["boom"], "只交 <patch>")
    assert turns == [{"role": "assistant", "content": "prev"}, {"role": "user", "content": "只交 <patch>"}]
    # An instruction is reason enough; an output to repair is still required.
    assert compose_repair_turns("prev", [], "只交 <patch>")[1]["content"] == "只交 <patch>"
    assert compose_repair_turns("", ["boom"], "只交 <patch>") == []


def test_an_escaped_patch_is_refused_rather_than_laid() -> None:
    """Laid over a base with real Chinese the whole no longer looks escaped, so the validator alone would let the
    literal escapes into the subtitles (owner 2026-09-04: a loud failure beats a quiet one)."""

    window = _window(4)
    _, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)
    escaped = _patch([r"sub|3|\u6e90|\u597d\u7684|high|"])
    merge = apply_patch(base, escaped)
    assert merge.mode == "invalid" and "literal escapes" in merge.errors[0]
    assert apply_patch(base, _patch([_sub("3", "好的")])).mode == "patch"


def test_a_leaked_reasoning_reply_says_why_first() -> None:
    from finesub.llm.output_protocol import LEAKED_THOUGHT_ERROR

    window = _window(4)
    _, base = _base(_answer([_sub("1"), _sub("2"), _sub("4")]), window)
    merge = apply_patch(base, "thought\nSource 3 is missing, so I will write it.")
    assert merge.mode == "invalid" and merge.errors[0] == LEAKED_THOUGHT_ERROR


def test_a_grouped_window_repairs_in_its_own_six_columns_whatever_variant_the_caller_holds() -> None:
    six = "type|position|corrected_text|translation|conf|note"
    from dataclasses import replace

    window = replace(_window(3, {1: Seam.NONE, 2: Seam.MAY, 3: Seam.MAY}), grouped=True)
    text = _answer(["sub|1|源1|译文1|high|", "sub|2|源2|译文2|high|a|b", "sub|3|源3|译文3|high|"], header=six)
    _result, base = _base(text, window)  # the cell's capableC: validated as capableC-grouped

    assert base is not None and base.header == six and base.pending == ("2",)
    assert f"`{six}`" in patch_instruction(base, ["Row 2 has 7 cells"])
    merged = apply_patch(base, _patch(["sub|2|源2|译文2|high|a｜b"], header=six))
    assert merged.mode == "patch" and validate_correction_window_output(merged.text, window, variant=CAPABLE_C).ok


@pytest.mark.parametrize(("row", "named"), [
    ("sub|2|1.5|0.5|源2|译文2|high|3|", "unexpected duration, gap and char_count columns"),
    ("sub|2|2.0|1.5|0.5|源2|译文2|high|3|", "unexpected start, duration, gap and char_count columns"),
])
def test_a_grouped_row_in_the_old_columns_is_its_own_error_and_still_gets_a_patch(row: str, named: str) -> None:
    from dataclasses import replace

    six = "type|position|corrected_text|translation|conf|note"
    window = replace(_window(3, {1: Seam.NONE, 2: Seam.MAY, 3: Seam.MAY}), grouped=True)
    text = _answer(["sub|1|源1|译文1|high|", row, "sub|3|源3|译文3|high|"], header=six)
    result, base = _base(text, window)

    [bad] = [verdict for verdict in result.rows if not verdict.ok]
    assert len(bad.errors) == 1 and named in bad.errors[0]
    assert [error for error in result.errors if named in error or "cells" in error] == list(bad.errors)
    assert base is not None and base.pending == ("2",)


def test_a_polish_set_patches_a_filler_line_with_its_polished_cell_empty() -> None:
    """A Chinese source's set: the patch is headed `polished`, and a filler line's empty cell passes."""
    polish = resolve_variant("capableC-polish")
    window = _window(4)
    header = correction_output_header(polish)
    assert "|polished|" in header

    text = _answer([_sub("1"), _sub("2"), _sub("4")], header=header)
    _result, base = _base(text, window, polish)
    assert base is not None and base.header == header and base.pending == ("3",)
    merged = apply_patch(base, _patch(["sub|3|呃||high|"], header=header))
    result = validate_correction_window_output(merged.text, window, variant=polish)
    assert result.ok, result.errors
    assert [segment.translation for segment in result.segments] == ["译文1", "译文2", "", "译文4"]


def test_a_placeholder_answer_is_rewritten_not_patched() -> None:
    # Every row passes its own checks; the window as a whole is refused (`output_protocol._with_placeholder`).
    window = _window()
    _result, base = _base(_answer([f"sub|{index}|text|text|high|" for index in range(1, 7)]), window)
    assert base is None
