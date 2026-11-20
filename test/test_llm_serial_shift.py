"""A correction answer whose rows slid off their sources (`serial_shift`), and how it goes back for repair."""

from __future__ import annotations

from finesub.llm.agent.agent_validators import build_validator, serialize_window
from finesub.llm.chunking import CorrectionBudget, SubtitleSegment, SubtitleWindow
from finesub.llm.correction_patch import apply_patch, repair_base
from finesub.llm.output_protocol import correction_output_header, validate_correction_window_output
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.serial_shift import SHIFT_FLOOR, SHIFT_MARGIN, SHIFT_SAME_OFFSET, find_shifts, normalize

CAPABLE_C = resolve_variant("capableC")
HEADER = correction_output_header(CAPABLE_C)


def _line(index: int) -> str:
    """Words no other line shares a character with."""
    return "".join(chr(0x4E00 + 40 * index + k) for k in range(6))


def _sources(count: int) -> dict[str, str]:
    return {str(index): _line(index) for index in range(1, count + 1)}


def _rows(texts: list[str]) -> list[tuple[tuple[str, ...], str]]:
    return [((str(index),), text) for index, text in enumerate(texts, start=1)]


def find_shift(rows, sources):
    """The first shift, or None."""
    shifts = find_shifts(rows, sources)
    return shifts[0] if shifts else None


def _shifted(count: int, start: int, length: int, step: int = 1) -> list[str]:
    """Rows from ``start`` on carry the words ``step`` sources further down, for ``length`` rows."""
    return [_line(i + step) if start <= i < start + length else _line(i) for i in range(1, count + 1)]


def test_rows_carrying_the_next_source_are_a_shift_from_where_it_starts() -> None:
    shift = find_shift(_rows(_shifted(40, 12, 20)), _sources(40))

    assert shift is not None
    assert (shift.start, shift.match, shift.sample, shift.rows) == ("12", "13", _line(13), 20)
    assert shift.sources == tuple(str(i) for i in range(12, 32))


def test_offsets_that_grow_and_rows_that_rotate_are_shifts_too() -> None:
    # one source skipped at 5, another at 15
    growing = [_line(i) for i in range(1, 5)] + [_line(i + 1) for i in range(5, 15)] + [_line(i + 2) for i in range(15, 25)]
    growing += [_line(i) for i in range(25, 41)]
    assert find_shift(_rows(growing), _sources(40)) is not None
    rotated = [_line(i) for i in range(1, 41)]
    rotated[19:24] = [rotated[23], *rotated[19:23]]
    shift = find_shift(_rows(rotated), _sources(40))
    assert shift is not None and shift.start == "20" and shift.rows == 5



def test_rows_that_skipped_a_long_stretch_are_a_shift_however_far() -> None:
    texts = [_line(i + 40) if 10 <= i < 20 else _line(i) for i in range(1, 81)]
    shift = find_shift(_rows(texts), _sources(80))
    assert shift is not None and (shift.start, shift.match, shift.rows) == ("10", "50", 10)


def test_rows_that_all_read_like_one_source_are_not_a_shift() -> None:
    # A recurring word corrected to how the ASR wrote it once, far off: one source, not a slide.
    texts = [_line(60) if 10 <= i < 18 else _line(i) for i in range(1, 81)]
    assert find_shift(_rows(texts), _sources(80)) is None

def test_two_rows_off_by_the_same_offset_are_a_shift_one_is_not() -> None:
    assert SHIFT_SAME_OFFSET == 2
    assert find_shift(_rows(_shifted(40, 10, 1)), _sources(40)) is None
    shift = find_shift(_rows(_shifted(40, 10, 2)), _sources(40))
    assert shift is not None and (shift.start, shift.rows, shift.sources) == ("10", 2, ("10", "11"))
    # Two rows off by different offsets are not.
    texts = [_line(i) for i in range(1, 41)]
    texts[9:11] = [_line(11), _line(14)]  # row 10 carries 11 (offset 1), row 11 carries 14 (offset 3)
    assert find_shift(_rows(texts), _sources(40)) is None
    # Nor are two single rows off by one with a row in its place between them.
    texts = _shifted(40, 10, 1)
    texts[11] = _line(13)
    assert find_shift(_rows(texts), _sources(40)) is None


def test_a_line_said_twice_corrected_to_its_other_copy_is_not_a_shift() -> None:
    # Two lines said again 20 sources later, the ASR mishearing one character of each the second time; the model
    # corrects both to the first copy, so each row reads like a source 20 back (the same offset).
    sources = _sources(40)
    sources["30"], sources["31"] = sources["10"][:-1] + "了", sources["11"][:-1] + "了"
    texts = [sources[str(i)] for i in range(1, 41)]
    texts[29:31] = [sources["10"], sources["11"]]
    assert find_shift(_rows(texts), sources) is None
    # A slide over other lines in the same run is still one: the alike rows only count toward no offset.
    texts[31:33] = [sources["33"], sources["34"]]
    shift = find_shift(_rows(texts), sources)
    assert shift is not None and (shift.start, shift.rows) == ("30", 4)


def test_rewrites_of_a_row_alone_are_not_a_shift() -> None:
    sources = _sources(40)
    # Kana rewritten, full-width digits, punctuation: the same words.
    sources.update({str(i): f"おねとんとん{i}かい" for i in range(10, 20)})
    texts = [_line(i) for i in range(1, 41)]
    texts[9:19] = [f"おねトントン、{i}かい！" for i in range(10, 20)]
    assert normalize("おねトントン、１０かい！") == normalize("おねとんとん10かい")
    assert find_shift(_rows(texts), sources) is None
    # A row rewritten past recognition reads like no source at all, its neighbours included.
    texts = [_line(i) for i in range(1, 41)]
    texts[9:19] = ["".join(chr(0x3400 + 13 * i + k) for k in range(5)) for i in range(10, 20)]
    assert find_shift(_rows(texts), _sources(40)) is None


def test_legal_merges_and_discards_are_not_a_shift() -> None:
    rows = [(("1",), _line(1))]
    rows += [((str(i), str(i + 1)), _line(i) + _line(i + 1)) for i in range(2, 30, 2)]
    rows += [((str(i),), _line(i)) for i in range(35, 41)]  # 30-34 discarded
    assert find_shift(rows, _sources(40)) is None


def _window(count: int) -> SubtitleWindow:
    segments = [SubtitleSegment(f"s{100 + i}", i * 2.0, i * 2.0 + 1.5, _line(i)) for i in range(1, count + 1)]
    return SubtitleWindow("0001", segments, [], "test", CorrectionBudget(0, 0, 0, ""), length_scale=1.0)


def _answer(texts: list[str]) -> str:
    rows = [f"sub|{i}|{text}|译文{i}|high|" for i, text in enumerate(texts, start=1)]
    return "<reasoning>\n全局判断\n</reasoning>\n<translated>\n" + "\n".join([HEADER, *rows]) + "\n</translated>\n"


def test_a_shift_fails_validation_and_says_where_it_starts() -> None:
    window = _window(40)
    texts = _shifted(40, 33, 7)  # the last row repeats nothing: it carries "source 41", which does not exist
    texts[-1] = _line(40)
    result = validate_correction_window_output(_answer(texts), window, variant=CAPABLE_C)

    assert not result.ok
    assert any("从源 33 起" in error and "源 34 的话" in error for error in result.errors)
    # What the start source itself says, and how to drop it rather than skip it.
    assert any(f"源 33 自己的话是「{_line(33)}」" in error and "discard|33" in error for error in result.errors)
    assert result.shifted_sources == tuple(f"s{100 + i}" for i in range(33, 40))
    assert all(verdict.ok for verdict in result.rows)  # the error is the window's, not a row's


def test_a_local_shift_goes_back_as_a_patch_of_its_rows() -> None:
    window = _window(40)
    texts = _shifted(40, 20, 6)
    result = validate_correction_window_output(_answer(texts), window, variant=CAPABLE_C)
    base = repair_base(_answer(texts), result, window, CAPABLE_C)

    assert base is not None and base.pending == tuple(str(i) for i in range(20, 26))
    patch = "<patch>\n" + "\n".join([HEADER, *(f"sub|{i}|{_line(i)}|译文{i}|high|" for i in range(20, 26))]) + "\n</patch>"
    merged = apply_patch(base, patch)
    assert merged.mode == "patch"
    assert validate_correction_window_output(merged.text, window, variant=CAPABLE_C).ok


def test_a_shift_over_half_the_window_goes_back_whole() -> None:
    window = _window(40)
    texts = _shifted(40, 10, 31)
    texts = texts[:39]  # the tail source is missing, as in the real ones
    result = validate_correction_window_output(_answer(texts), window, variant=CAPABLE_C)

    assert any("missing source" in error for error in result.errors)
    assert repair_base(_answer(texts), result, window, CAPABLE_C) is None


def test_shifts_at_either_end_or_backwards_are_found_where_they_start() -> None:
    backwards = [_line(i - 1) if 20 <= i < 26 else _line(i) for i in range(1, 41)]
    assert find_shift(_rows(backwards), _sources(40)).start == "20"
    assert find_shift(_rows(_shifted(40, 1, 8)), _sources(40)).start == "1"
    to_the_end = _shifted(40, 30, 10)[:39]  # rows 30-39 carry 31-40, source 40 has no row
    shift = find_shift(_rows(to_the_end), _sources(40))
    assert shift is not None and shift.sources == tuple(str(i) for i in range(30, 40))


def test_two_shifts_far_apart_go_back_on_their_own() -> None:
    texts = _shifted(300, 20, 6)
    texts[249:256] = [_line(i + 1) for i in range(250, 257)]
    shifts = find_shifts(_rows(texts), _sources(300))
    assert [(shift.start, shift.sources[-1]) for shift in shifts] == [("20", "25"), ("250", "256")]

    window = _window(300)
    result = validate_correction_window_output(_answer(texts), window, variant=CAPABLE_C)
    assert "从源 20 起" in result.errors[-1] and "从源 250 起" in result.errors[-1]  # one error naming both
    base = repair_base(_answer(texts), result, window, CAPABLE_C)
    # Only the two slides go back; the rows between them stay as they were.
    assert base is not None and base.pending == (*(str(i) for i in range(20, 26)), *(str(i) for i in range(250, 257)))


def test_a_shift_and_a_broken_row_inside_it_go_back_together() -> None:
    window = _window(40)
    texts = _shifted(40, 20, 6)
    answer = _answer(texts).replace(f"sub|22|{texts[21]}|译文22|high|", "sub|22||译文22|high|")
    result = validate_correction_window_output(answer, window, variant=CAPABLE_C)
    # The broken row is no row of the shift, nor does it cut the run in two.
    assert result.shifted_sources == ("s120", "s121", "s123", "s124", "s125")
    assert "Row 22 has empty corrected text." in result.errors

    base = repair_base(answer, result, window, CAPABLE_C)
    assert base is not None and base.pending == tuple(str(i) for i in range(20, 26))


def test_a_tool_session_gets_a_local_shift_back_as_a_patch() -> None:
    window = _window(40)
    manifest = {"metadata": {"validator": {"window": serialize_window(window)}, "variant": "capableC",
                             "capability_tier": "capable"}}
    verdict = build_validator("correction-window", manifest)(_answer(_shifted(40, 20, 6)), {})

    assert verdict.status == "repairable" and any("从源 20 起" in error for error in verdict.errors)
    assert "<patch>" in verdict.metadata["repair_instruction"]
    assert verdict.metadata["rows_to_fix"].splitlines()[0].startswith("sub|20|")


def _letters(start: int, count: int) -> str:
    return "".join(chr(0x4E00 + start + k) for k in range(count))


def test_the_floor_and_the_margin_are_inclusive_and_strict_below() -> None:
    """Each shifted row is 4 characters; the source it carries shares 2 of them (ratio 0.5, the floor)."""

    def answer(own_shared: int, other_shared: int) -> tuple[list[tuple[tuple[str, ...], str]], dict[str, str]]:
        sources = {str(i): _letters(1000 + 10 * i, 4) for i in range(1, 41)}
        rows = []
        for i in range(1, 41):
            if 10 <= i < 15:
                written = _letters(5000 + 10 * i, 4)
                sources[str(i)] = written[:own_shared] + _letters(6000 + 10 * i, 4 - own_shared)
                sources[str(i + 20)] = written[:other_shared] + _letters(7000 + 10 * i, 4 - other_shared)
                rows.append(((str(i),), written))
            else:
                rows.append(((str(i),), sources[str(i)]))
        return rows, sources

    assert find_shift(*answer(own_shared=0, other_shared=2)) is not None  # 0.5 against 0: at the floor
    assert find_shift(*answer(own_shared=0, other_shared=1)) is None  # 0.25: below the floor
    assert find_shift(*answer(own_shared=2, other_shared=3)) is not None  # 0.75 against 0.5: past the margin
    assert find_shift(*answer(own_shared=2, other_shared=2)) is None  # 0.5 against 0.5: short of the margin


def test_rows_reading_like_far_sources_are_a_shift_only_when_two_share_an_offset() -> None:
    texts = [_line(i) for i in range(1, 81)]
    texts[9:13] = [_line(60), _line(50), _line(70), _line(55)]  # rows 10-13: offsets 50, 39, 58, 42
    assert find_shift(_rows(texts), _sources(80)) is None
    texts[9:13] = [_line(60), _line(61), _line(70), _line(55)]  # rows 10 and 11 both 50 further down
    assert find_shift(_rows(texts), _sources(80)) is not None


def test_skipping_hopeless_comparisons_flags_nothing_full_comparison_would_not() -> None:
    """Where comparing every row with every source in full finds no two off rows in a row, find_shifts finds
    no shift either (the other direction, and the start, were checked equal on every recorded answer)."""
    import random
    from difflib import SequenceMatcher

    def reference(rows, sources):
        order = list(sources)
        text = {k: normalize(v) for k, v in sources.items()}
        flags = []
        for ids, corrected in rows:
            written = normalize(corrected)
            own = SequenceMatcher(None, written, "".join(text[i] for i in ids), autojunk=False).ratio()
            best = max((SequenceMatcher(None, written, text[k], autojunk=False).ratio()
                        for k in order if k not in ids and text[k]), default=0.0)
            flags.append(best >= max(SHIFT_FLOOR, own + SHIFT_MARGIN))
        run = longest = 0
        for flag in flags:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return longest

    rng = random.Random(7)
    alphabet = "".join(chr(0x3041 + k) for k in range(12))
    for _ in range(60):
        sources = {str(i): "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 9))) for i in range(1, 31)}
        texts = [sources[str(i)] for i in range(1, 31)]
        start, step = rng.randrange(1, 20), rng.choice([1, 2, 7, 15])
        texts[start:] = [sources[str(min(30, i + step))] for i in range(start + 1, 31)]
        rows = _rows(texts)
        shift = find_shift(rows, sources)
        if reference(rows, sources) < SHIFT_SAME_OFFSET:
            assert shift is None


def test_a_copied_duration_cell_is_named_not_just_counted() -> None:
    basic = resolve_variant("basicA")
    header = correction_output_header(basic)
    window = _window(6)
    rows = [f"sub|{i}|{i * 2.0}|1.5|{_line(i)}|译文{i}|high|" for i in range(1, 7)]  # duration after start
    answer = "<reasoning>\n全局判断\n</reasoning>\n<translated>\n" + "\n".join([header, *rows]) + "\n</translated>\n"
    errors = validate_correction_window_output(answer, window, variant=basic).errors

    assert any("extra cell after start (`1.5`, the input's duration copied over" in error for error in errors)
