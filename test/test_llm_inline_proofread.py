"""The correction window's own proofread turn (`inline_proofread`): the reply, the instruction and the flags."""

from __future__ import annotations

from finesub.llm import inline_proofread
from finesub.llm.chunking import SubtitleSegment, SubtitleWindow
from finesub.llm.output_protocol import correction_output_header, validate_correction_window_output
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.stages.proofread.prompt import criteria, system_prompt

CAPABLE_C = resolve_variant("capableC")
HEADER = correction_output_header(CAPABLE_C)


def _window(count: int = 4) -> SubtitleWindow:
    segments = [SubtitleSegment(f"s{100 + index}", index * 2.0, index * 2.0 + 1.5, f"原文{index}")
                for index in range(1, count + 1)]
    return SubtitleWindow("0001", segments, [], "test", None, length_scale=1.0)  # type: ignore[arg-type]


def _sub(position: str, translation: str | None = None, corrected: str | None = None, note: str = "") -> str:
    return f"sub|{position}|{corrected or '源' + position}|{translation or '译文' + position}|high|{note}"


def _draft() -> str:
    return "<translated>\n" + "\n".join([HEADER, _sub("1,2"), _sub("3"), _sub("4")]) + "\n</translated>\n"


def _patch(rows: list[str]) -> str:
    return "<patch>\n" + "\n".join([HEADER, *rows]) + "\n</patch>"


def _apply(reply: str):
    window = _window()
    return inline_proofread.apply(_draft(), reply, window, CAPABLE_C,
                                  lambda text: validate_correction_window_output(text, window, variant=CAPABLE_C))


def test_a_text_patch_is_laid_over_the_draft_and_its_changes_are_listed() -> None:
    outcome = _apply(_patch([_sub("3", "改过的译文", note="1：漏译了否定")]))

    assert outcome.status == inline_proofread.APPLIED
    assert "sub|3|源3|改过的译文|high|1：漏译了否定" in outcome.text and "sub|1,2|源1,2|译文1,2|high|" in outcome.text
    assert [(c.position, c.before[1], c.after[1], c.note) for c in outcome.changes] == [
        ("3", "译文3", "改过的译文", "1：漏译了否定")]


def test_a_header_only_patch_changes_nothing() -> None:
    outcome = _apply(_patch([]))
    assert outcome.status == inline_proofread.NO_CHANGE and outcome.text == _draft()


def test_a_patch_that_regroups_or_discards_keeps_the_draft() -> None:
    for rows in ([_sub("1")], [_sub("3,4")], ["discard|4"]):
        outcome = _apply(_patch(rows))
        assert outcome.status == inline_proofread.FALLBACK and outcome.text == _draft(), rows
        assert outcome.errors


def test_a_reply_without_a_patch_keeps_the_draft() -> None:
    outcome = _apply("<translated>\n" + HEADER + "\n</translated>")
    assert outcome.status == inline_proofread.FALLBACK and outcome.text == _draft()


def test_a_patch_that_empties_a_text_cell_keeps_the_draft() -> None:
    outcome = _apply(_patch([_sub("3", " ")]))
    assert outcome.status == inline_proofread.FALLBACK and outcome.text == _draft()


def test_a_patch_the_whole_answer_fails_with_keeps_the_draft() -> None:
    outcome = _apply(_patch([_sub("3", "译|文")]))
    assert outcome.status == inline_proofread.FALLBACK and outcome.text == _draft() and outcome.errors


def test_the_instruction_carries_pass_3s_criteria_and_the_flags() -> None:
    text = inline_proofread.instruction(CAPABLE_C, _window(), ["- 3（读速）：跨度 1.0 秒、12 个字，12.0 字/秒"])
    assert criteria() in text and criteria() in system_prompt()
    assert "- 3（读速）" in text and HEADER in text and "$" not in text
    assert "（这一窗没有点名的行）" in inline_proofread.instruction(CAPABLE_C, _window(), [])


def test_flags_read_speed_and_a_prepared_name_missing_from_the_translation() -> None:
    window = _window()
    draft = "<translated>\n" + "\n".join([
        HEADER, _sub("1,2", "这一行的译文有整整十二个字"), _sub("3", "他来了", corrected="タルタリヤが来た"), _sub("4")]) + "\n</translated>\n"
    result = validate_correction_window_output(draft, window, variant=CAPABLE_C)
    names = inline_proofread.names_table({"entities": [{"name": "タルタリヤ", "zh": "达达利亚／公子"}]})

    flagged = inline_proofread.flagged_rows(result, window, names)

    assert [line.split("（")[0] for line in flagged] == ["- 3"]
    assert "タルタリヤ" in flagged[0]


def test_a_reply_that_cannot_be_laid_over_gets_a_repair_round_whose_message_carries_its_errors() -> None:
    outcome = _apply(_patch([_sub("3", "译|文")]))
    assert outcome.repairable
    text = inline_proofread.repair_instruction(CAPABLE_C, _window(), outcome.errors)
    assert all(error in text for error in outcome.errors) and HEADER in text and "$" not in text
    assert not _apply(_patch([])).repairable and not _apply(_patch([_sub("3", "改过的译文")])).repairable


def test_a_draft_that_does_not_pass_is_not_repaired() -> None:
    window = _window()
    outcome = inline_proofread.apply("<translated>\n" + HEADER + "\n</translated>\n", _patch([_sub("3")]), window,
                                     CAPABLE_C, lambda text: validate_correction_window_output(text, window,
                                                                                               variant=CAPABLE_C))
    assert outcome.status == inline_proofread.FALLBACK and not outcome.repairable


def test_on_a_tracked_window_a_patch_that_changes_spk_or_conf_keeps_the_draft(tmp_path) -> None:
    """The proofread turn on a `capableC-tracked` answer: `spk` is copied like `conf`; the validator alone would
    let another legal speaker through."""
    from .test_llm_tracked_single_pass import HEADER as TRACKED_HEADER, TRACKED, _stable, _window as tracked_window

    variant, window = resolve_variant(TRACKED), tracked_window(_stable(tmp_path))
    row = {1: "sub|1|s0|a|甲|high|", 2: "sub|2|s0|b|乙|high|", 3: "sub|3|s1|c|丙|high|", 4: "sub|4|s1|d|丁|high|"}
    draft = "<translated>\n" + "\n".join([TRACKED_HEADER, *row.values()]) + "\n</translated>\n"

    def apply(rows: list[str]):
        reply = "<patch>\n" + "\n".join([TRACKED_HEADER, *rows]) + "\n</patch>"
        return inline_proofread.apply(draft, reply, window, variant,
                                      lambda text: validate_correction_window_output(text, window, variant=variant))

    assert apply(["sub|3|s1|c|改过的丙|high|1：漏译"]).status == inline_proofread.APPLIED
    for changed, column in (("sub|3|s0|c|改过的丙|high|1：漏译", "spk"), ("sub|3|@佐藤|c|丙|high|", "spk"),
                            ("sub|3|s1|c|改过的丙|low|1：漏译", "conf")):
        outcome = apply([changed])
        assert outcome.status == inline_proofread.FALLBACK and outcome.text == draft and outcome.repairable
        assert any(f"changes `{column}`" in error for error in outcome.errors), outcome.errors
    text = inline_proofread.instruction(variant, window, [])
    assert "`spk`、`conf` 照抄" in text and TRACKED_HEADER in text and "$" not in text
    assert "说话人这一轮不重判" in text
    assert "`spk`、`conf` 也照抄" in inline_proofread.repair_instruction(variant, window, ["x"])
    untracked = inline_proofread.instruction(CAPABLE_C, _window(), [])
    assert "，`conf` 照抄，" in untracked and "不重判" not in untracked


def test_a_patch_that_slides_the_translations_off_their_rows_keeps_the_draft() -> None:
    """`translation_shift` only warns, so a patch that made one would pass: the turn hands back the draft instead,
    with no repair round (owner 2026-11-11). A draft that already had the shift is not held to it."""
    from .test_llm_translation_shift import _answer, _slid, _source, _window as shift_window

    count, window = 40, shift_window(40)

    def validate(text: str):
        return validate_correction_window_output(text, window, variant=CAPABLE_C)

    def patch(translations: list[str]) -> str:
        rows = [f"sub|{i}|{_source(i)}|{translations[i - 1]}|high|" for i in range(11, 21)]
        return _patch(rows)

    clean, slid = _answer(CAPABLE_C, _slid(count, 1, 0)), _slid(count, 11, 10)
    outcome = inline_proofread.apply(clean, patch(slid), window, CAPABLE_C, validate)
    assert outcome.status == inline_proofread.FALLBACK and outcome.text == clean and not outcome.repairable
    assert outcome.errors[0].startswith("the patch made the translations slide") and "s111–s120" in outcome.errors[1]
    follow, repair = inline_proofread.next_step(clean, patch(slid), window, CAPABLE_C, validate, round_=0)
    assert follow.status == inline_proofread.FALLBACK and repair == ""
    # the draft already slid: a patch over it that keeps the shift is not what made it
    shifted = _answer(CAPABLE_C, slid)
    kept = inline_proofread.apply(shifted, patch(slid[:10] + ["改过的" + text for text in slid[10:20]] + slid[20:]),
                                  window, CAPABLE_C, validate)
    assert validate(shifted).translation_shift is not None and kept.status == inline_proofread.APPLIED
