"""Game-glossary matching: verbatim forms and the sound matcher's fixed thresholds."""

from __future__ import annotations

import pytest

from finesub.llm import glossary_match as gm


def test_verbatim_longer_forms_mask_shorter_and_short_katakana_needs_an_edge():
    forms = gm.by_length(["クレー", "リン", "リンネア", "鍾離"])
    assert gm.verbatim("クレープとリンネア", forms) == {"リンネア"}
    assert gm.verbatim("クレーと鍾離", forms) == {"クレー", "鍾離"}
    assert gm.verbatim("クレーーが", forms) == {"クレー"}  # one lengthened vowel is still the name
    assert gm.verbatim("クレーンが", forms) == set()


def test_phonemes_read_long_vowels_only_in_readings():
    assert gm.phonemes("テイワット") == ("t", "e", "i", "w", "a", "Q", "t", "o")
    assert gm.phonemes("ショウリ", reading=True) == ("sy", "o", ":", "r", "i")


@pytest.fixture(scope="module")
def matcher():
    for name in ("pyopenjtalk", "sudachipy", "rapidfuzz"):
        pytest.importorskip(name)
    terms = [gm.Term("テイワット", "テイワット", False), gm.Term("オデット", "オデット", False),
             gm.Term("セージ", "セージ", False), gm.Term("ローエン", "ローエン", False),
             gm.Term("鍾離", "ショウリ", True)]
    return gm.SoundMatcher(terms)


def _hits(matcher, text, frequent=frozenset()):
    return {h.ja: h.surface for h in matcher.match([text], frequent)[text]}


@pytest.mark.parametrize("text, ja, surface", [
    ("ヘイワットの冒険", "テイワット", "ヘイワット"),  # first consonant heard as h
    ("おでっとだよね", "オデット", "おでっと"),       # a katakana name written in hiragana
    ("老苑さんが来た", "ローエン", "老苑"),           # a katakana name written as kanji
    ("勝利が言ってた", "鍾離", "勝利"),
])
def test_sound_hits(matcher, text, ja, surface):
    assert _hits(matcher, text).get(ja) == surface


def test_sound_misses(matcher):
    assert "セージ" not in _hits(matcher, "次のページを見て")
    assert _hits(matcher, "テイワットとヘイワット") == {}       # the name is already in the row


def test_noisy_pairs_are_dropped():
    for name in ("pyopenjtalk", "sudachipy", "rapidfuzz"):
        pytest.importorskip(name)
    terms = [gm.Term("テイワット", "テイワット", False)]
    text = "ヘイワットの冒険"
    assert gm.SoundMatcher(terms, noise={("テイワット", "ヘイワット"): 3e-4}).match([text])[text][0].noise == 3e-4
    assert gm.SoundMatcher(terms, noise={("テイワット", "ヘイワット"): 4e-4}).match([text])[text] == []


def test_frequent_terms_pass_a_looser_line(matcher):
    text = "メイワットに行く"  # m for t: far, about 0.11
    assert "テイワット" not in _hits(matcher, text)
    assert _hits(matcher, text, frozenset({"テイワット"})).get("テイワット") == "メイワット"
