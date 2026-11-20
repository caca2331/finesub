"""Translations that slid off their rows while the corrected text stayed in place: a warning, never a rejection
(`finesub.llm.translation_shift`, docs/llm_harness_behavior.md 「重试与拼接」)."""

from __future__ import annotations

import json

from finesub.llm.chunking import CorrectionBudget, SubtitleSegment, SubtitleWindow
from finesub.llm.client import LLMCallResult
from finesub.llm.output_protocol import correction_output_header, validate_correction_window_output
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.routing.config import LLMRole
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.translation_shift import MIN_GAINING, MIN_ROWS, find_translation_shift
from finesub.reporting import reporting_to

from .conftest import setattr_correction
from .support.fakes import FakeTokenCounter, WarningRecorder


CAPABLE_C = resolve_variant("capableC")


def _kanji(index: int) -> str:
    """Two Han characters no other line shares."""
    return chr(0x4E00 + 7 * index) + chr(0x4E00 + 7 * index + 3)


def _source(index: int) -> str:
    return _kanji(index) + "です"


def _translation(index: int) -> str:
    return _kanji(index) + "啊。"


def _sources(count: int) -> dict[str, str]:
    return {str(index): _source(index) for index in range(1, count + 1)}


def _rows(translations: list[str]) -> list[tuple[tuple[str, ...], str]]:
    return [((str(index),), text) for index, text in enumerate(translations, start=1)]


def _slid(count: int, start: int, length: int, step: int = 1) -> list[str]:
    """Rows from ``start`` on carry the translation of the row ``step`` further down, for ``length`` rows."""
    return [_translation(i + step) if start <= i < start + length else _translation(i) for i in range(1, count + 1)]


def test_translations_one_row_off_are_flagged_from_where_they_start() -> None:
    found = find_translation_shift(_rows(_slid(40, 11, 10)), _sources(40))

    assert found is not None
    assert (found.start, found.end, found.rows, found.offset) == ("11", "20", 10, 1)
    assert find_translation_shift(_rows(_slid(40, 11, 6, step=-2)), _sources(40)).offset == -2
    assert find_translation_shift(_rows(_slid(40, 1, 40)[:40]), _sources(40)) is not None


def test_clean_translations_are_not_flagged() -> None:
    assert find_translation_shift(_rows(_slid(40, 1, 0)), _sources(40)) is None


def test_fewer_rows_than_the_minimum_off_are_not_flagged() -> None:
    assert find_translation_shift(_rows(_slid(40, 11, MIN_GAINING)), _sources(40)) is not None
    assert find_translation_shift(_rows(_slid(40, 11, MIN_GAINING - 1)), _sources(40)) is None


def test_translations_swapped_for_word_order_are_not_flagged() -> None:
    translations = _slid(40, 1, 0)
    for first in (5, 9, 13, 17, 21, 25):  # six swapped pairs, close together
        translations[first - 1], translations[first] = translations[first], translations[first - 1]
    assert find_translation_shift(_rows(translations), _sources(40)) is None


def test_rows_whose_sources_read_alike_gain_nothing() -> None:
    sources = _sources(40)
    translations = _slid(40, 1, 0)
    for index in range(11, 22):  # lines differing only in a number: one row's sources cannot be told from the next
        sources[str(index)] = f"第{index}回です"
    for index in range(11, 21):
        translations[index - 1] = f"第{index + 1}回。"  # one row off, which only the number shows
    assert find_translation_shift(_rows(translations), sources) is None
    for index in range(11, 22):  # the same slide over lines that differ
        sources[str(index)] = f"{_kanji(index)}{_kanji(index + 60)}{_kanji(index + 120)}{index}"
    for index in range(11, 21):
        translations[index - 1] = f"{_kanji(index + 1)}{_kanji(index + 61)}{_kanji(index + 121)}{index + 1}"
    assert find_translation_shift(_rows(translations), sources) is not None


def test_short_windows_and_rows_without_translation_are_not_judged() -> None:
    count = MIN_ROWS - 1
    assert find_translation_shift(_rows(_slid(count, 2, 10)), _sources(count)) is None
    # Rows without translation are skipped, not counted, and do not break a stretch.
    translations = _slid(40, 11, 10)
    translations[14] = ""
    found = find_translation_shift(_rows(translations), _sources(40))
    assert (found.start, found.end, found.rows) == ("11", "20", 9)


def _window(count: int) -> SubtitleWindow:
    segments = [SubtitleSegment(f"s{100 + i}", i * 2.0, i * 2.0 + 1.5, _source(i)) for i in range(1, count + 1)]
    return SubtitleWindow("0001", segments, [], "test", CorrectionBudget(0, 0, 0, ""), length_scale=1.0)


def _answer(variant, translations: list[str]) -> str:
    header = correction_output_header(variant)
    start = "|0.000" if variant.output_has_start else ""
    rows = [f"sub|{i}{start}|{_source(i)}|{text}|high|" for i, text in enumerate(translations, start=1)]
    return "<translated>\n" + "\n".join([header, *rows]) + "\n</translated>\n"


def test_a_capable_answer_passes_with_the_shift_in_source_ids() -> None:
    result = validate_correction_window_output(_answer(CAPABLE_C, _slid(40, 11, 10)), _window(40), variant=CAPABLE_C)

    assert result.ok and not result.errors
    assert (result.translation_shift.start, result.translation_shift.end) == ("s111", "s120")


def test_basic_a_and_grouped_answers_are_checked_too_correct_only_ones_not() -> None:
    for name in ("basicA", "capableC-grouped"):
        variant = resolve_variant(name)
        result = validate_correction_window_output(_answer(variant, _slid(40, 11, 10)), _window(40), variant=variant)
        assert result.ok and result.translation_shift.start == "s111", name
    # A correct-only set's second column is a polish, not a translation: not checked.
    polish = resolve_variant("capableC-polish")
    result = validate_correction_window_output(_answer(polish, _slid(40, 11, 10)), _window(40), variant=polish)
    assert result.ok and result.translation_shift is None


def test_an_accepted_shift_warns_and_lands_in_the_window_record(tmp_path, monkeypatch) -> None:
    count = 30
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"metadata": {}, "segments": [
        {"id": str(i), "start": i * 2.0, "end": i * 2.0 + 1.0, "text": _source(i)} for i in range(1, count + 1)
    ]}), encoding="utf-8")
    answer = _answer(CAPABLE_C, _slid(count, 6, 10))
    calls: list[list[str]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            calls.append(list(kwargs.get("validation_errors", ())))
            return LLMCallResult(content=answer, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                                 raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    warnings: list[tuple[str, str]] = []

    with reporting_to(WarningRecorder(warnings)):
        execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt",
                                   token_counter=FakeTokenCounter(), max_retries_per_window=1,
                                   max_replacements_per_window=0, profile=resolve_profile("text", "none", "quality"),
                                   task_artifact_dir=tmp_path / "artifacts", length_scale=1.0)
    assert calls == [[]]  # accepted as it came: no repair round
    assert any(code == "correction-translation-shift" and "源段 6–15" in message for code, message in warnings)
    ledger = [json.loads(line) for line in (tmp_path / "artifacts" / "correction-windows.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert ledger[-1]["translation_shift"]["start"] == "6"
    assert ledger[-1]["translation_shift"]["offset"] == 1
