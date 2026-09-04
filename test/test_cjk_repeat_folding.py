"""Repeat folding must actually fire on CJK, not merely exist.

An external engine shipped a repeat collapser that split words on ASCII
whitespace. Chinese and Japanese are one "word" to that tokenizer, so the guard
never fired on either -- dead code for months, in the two backends whose material
was most CJK, while `hey hey hey hey` folded fine and made it look alive.

This tree does not have that shape: `_text_unit_symbols` gives every non-space
script character its own symbol. These tests pin that, because "we checked once"
is not a property anyone can re-derive from the code six months from now, and a
tokenizer change would otherwise take the CJK path out silently.

The samples are the real ones: `謝謝觀看` is Whisper's signature Chinese
hallucination, and the kana forms are the everyday Japanese shapes.
"""

from __future__ import annotations

from finesub import text as asr_text


def _words(token: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "word": token,
            "start": index * 0.3,
            "end": index * 0.3 + 0.25,
            "space_before": False,
        }
        for index in range(count)
    ]


# --- within one token ------------------------------------------------------


def test_a_repeated_cjk_motif_inside_one_token_is_folded() -> None:
    folded, changed = asr_text.collapse_repeating_pattern("謝謝觀看，" * 9)
    assert changed
    assert len(folded) < len("謝謝觀看，" * 9)


def test_a_stretched_kana_run_is_folded() -> None:
    folded, changed = asr_text.collapse_repeating_pattern("あ" * 12)
    assert changed
    assert folded == "あ" * asr_text.REPEAT_KEEP_RUN


def test_ordinary_cjk_text_is_left_alone() -> None:
    """The negative control: folding must not touch real subtitles."""

    sentence = "这是一句正常的中文字幕不应该被折叠"
    folded, changed = asr_text.collapse_repeating_pattern(sentence)
    assert not changed
    assert folded == sentence


# --- across words ----------------------------------------------------------


def test_a_repeated_cjk_word_run_collapses_across_words() -> None:
    collapsed = asr_text.collapse_repeating_segment_words(_words("謝謝觀看，", 9))
    assert len(collapsed) < 9


def test_a_repeated_kana_word_run_collapses_across_words() -> None:
    collapsed = asr_text.collapse_repeating_segment_words(_words("はい、", 9))
    assert len(collapsed) < 9


def test_distinct_cjk_words_are_not_collapsed() -> None:
    words = [
        {"word": token, "start": i * 0.3, "end": i * 0.3 + 0.25, "space_before": False}
        for i, token in enumerate("今天天氣很好我們出去走走")
    ]
    assert len(asr_text.collapse_repeating_segment_words(words)) == len(words)


# --- group-level cycle -----------------------------------------------------


def test_the_group_cycle_detector_sees_a_cjk_motif() -> None:
    found = asr_text.detect_repeating_group_cycle("謝謝觀看，" * 12)
    assert found is not None


def test_a_short_cjk_motif_is_below_the_group_units_floor() -> None:
    """A documented boundary, not a defect.

    `はい、` twelve times is 24 word units and the group detector needs
    `GROUP_REPEAT_MIN_UNITS` (32) before it will call something a cycle -- the
    floor that keeps it off legitimate short repeats. The run is still caught
    one layer down, by the cross-word collapse above, which is why this is
    pinned as a boundary rather than filed as a gap.
    """

    text = "はい、" * 12
    assert asr_text.count_word_units(text) < asr_text.GROUP_REPEAT_MIN_UNITS
    assert asr_text.detect_repeating_group_cycle(text) is None
    assert len(asr_text.collapse_repeating_segment_words(_words("はい、", 12))) < 12


def test_word_units_count_cjk_characters_rather_than_whitespace_tokens() -> None:
    """The property the external engine lacked, stated directly.

    Under an ASCII-whitespace tokenizer this whole string is one token and every
    downstream count reads 1. Here each character carries its own unit, which is
    what keeps the CJK path alive.
    """

    assert asr_text.count_word_units("謝謝觀看") == 4.0
    assert asr_text.count_word_units("hello") == asr_text.SPACE_DELIMITED_WORD_UNITS
