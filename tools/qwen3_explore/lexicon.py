"""Tokens that essentially never begin a cue — shared by the splitter and the scorer.

Derived, not hand-written: `start_lift.py` measures each token's start-lift over the 3378 baseline
cues of all 11 clips and keeps those under 0.25 at >= 80 occurrences.

**There is one table per tokenization, and that is not an oversight.** Two wrong versions came
before it, in both directions:

1. One table, derived from the Whisper baseline's whisper-timestamped subwords, applied to the
   Qwen arm whose words are nagisa morphemes. 12 of its 26 entries are tokens nagisa essentially
   never emits (`して` `った` `り` `んだ`, all count 0), so those mid-word cuts were invisible on
   the Qwen arm — a systematic under-count in Qwen's favour.
2. Their union, applied to both arms. This looks symmetric and is not: a token earns its place
   from the tokenizer that observed it, and 10 of the 14 nagisa-only entries are ordinary word
   *openers* under whisper's subword inventory — 「か」 begins 「かわいい」, 「な」 begins
   「なに」, 「と」 begins 「とにかく」, and 「さ」 opens cues at 1.37× the base rate. Manual
   adjudication put the whisper arm's mid-word precision at 3/12 under the union against 7/12 for
   Qwen: the fix had merely reversed who was being flattered, and (worse) the splitter's
   `fragment_penalty` was steering the whisper arm away from perfectly good cuts.

So each arm is measured and optimised against the table derived from *its own* tokenizer. The
consequence is stated wherever the number appears: **the mid-word column is valid within an arm
and not comparable across arms.** `start_lift.py`'s cross-lift audit prints the 10 offenders.

Regenerate with:
    python -m tools.qwen3_explore.start_lift --out out/qwen-explore/cannot-start.json
"""

from __future__ import annotations

_SHARED = ("から", "が", "けど", "こと", "って", "と", "ない", "に", "の", "を")
_WHISPER_ONLY = ("き", "け", "した", "して", "っ", "った", "ら", "り", "る", "んだ", "ラ", "ン", "ー")
_NAGISA_ONLY = ("いう", "か", "さ", "し", "た", "て", "てる", "です", "な", "ね", "は", "まし", "も", "よ")

CANNOT_START_WHISPER = frozenset(_SHARED + _WHISPER_ONLY)
CANNOT_START_NAGISA = frozenset(_SHARED + _NAGISA_ONLY)

PUNCT_STRIP = "、。！？ 　,.!?…「」『』（）()"


def lexicon_for(words_carry_punct: bool) -> frozenset[str]:
    """Pick the table matching the word stream. `words_carry_punct` is the whisper-stream flag."""
    return CANNOT_START_WHISPER if words_carry_punct else CANNOT_START_NAGISA
