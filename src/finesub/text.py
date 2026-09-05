"""Text utility helpers for the ASR pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

from .reporting import current_reporter

#: One `\uXXXX` escape, as a CLI that cannot put a non-ASCII character into a
#: tool argument writes it. See `looks_escaped`.
_ESCAPED_CODEPOINT = re.compile(r"\\u([0-9a-fA-F]{4})")


def decodable_escape_count(value: str) -> int:
    """How many escapes in `value` `unescape_codepoints` would actually decode.

    Counts only the ones above U+007F, i.e. exactly the ones that get
    replaced. A plain match count would include escapes spelling ASCII
    characters -- which are deliberately left alone -- so reporting one
    would overstate what was changed, in a line whose whole job is to
    record what was changed.
    """

    return sum(
        1
        for digits in _ESCAPED_CODEPOINT.findall(value)
        if int(digits, 16) > 127
    )


def looks_escaped(value: str) -> bool:
    r"""Whether `value` reached us with its non-ASCII characters escaped.

    antigravity-cli 1.1.24 spells every non-ASCII codepoint of a `tools/call`
    argument as a literal `\uXXXX` *inside* the JSON string, so `json.loads`
    hands back the six characters of the escape instead of the character.
    Nothing else in the frame is escaped -- quotes and backslashes are not
    doubled -- so this is not a JSON string to parse twice; it is one
    transformation applied to the non-ASCII characters alone.

    Two conditions, and both are load-bearing:

    * **Pure ASCII.** The transformation leaves nothing else behind. (This is
      also the predicate's blind spot: a driver that escaped only *some*
      characters would not match. That gap is known and deliberate -- closing
      it means dropping this condition, which is a different trade.)
    * **At least one escape yields a non-ASCII character.** Text that meant to
      hold the six characters of an ASCII escape decodes to ASCII and is left
      alone.

    ⚠ There is deliberately **no floor on how many** (owner, 2026-09-04). Two
    earlier attempts at one are both gone, and the second is the instructive
    one:

    * A **density** floor read the *source language* rather than the fault --
      the same complete corruption measures 0.859 on a Japanese source and
      0.072 on an English one (`tools/escape_density/`).
    * A **count** floor of four read the *window size*. It was there to spare a
      legitimately ASCII reply carrying one literal escape, but it silently
      dropped the real fault whenever the reply was small: a one-row window
      translated into two characters arrives as two escapes and was waved
      through, delivering `\u597d\u7684` as a subtitle line.

    Both traded a **loud, rare** failure for a **silent** one, in a pipeline
    whose correction target is fixed Chinese (`PROMPT_VERSION`) -- so a
    legitimate reply is essentially never pure ASCII to begin with, and the
    false positive the floor was buying protection from is close to
    unreachable. The remaining exposure is one badly-chosen edit on a reply
    that is ASCII end to end, which is loud enough to notice and rare enough
    to accept.

    ⚠ **Judge whole frames, not single fields.** The fault is frame-shaped --
    the CLI escapes everything it sends -- while the evidence per field is
    not: one mostly-English note carrying two CJK characters yields two
    escapes on its own. Deciding per field would repair some cells of a reply
    and leave others escaped, which is worse than either answer. Callers pass
    the whole reply, or every argument of one call joined together.
    """

    if not value.isascii():
        return False
    return any(
        int(digits, 16) > 127 for digits in _ESCAPED_CODEPOINT.findall(value)
    )


def unescape_codepoints(value: str) -> str:
    r"""`value` with its `\uXXXX` escapes decoded where they stand.

    Decoding only -- `looks_escaped` is the judgement, and this assumes it has
    already been made about the frame this string belongs to. Surrogate pairs
    are rejoined, because a character outside the BMP arrives as two escapes
    and decoding them separately would leave an unpaired surrogate rather than
    the character. A lone surrogate gives the string back untouched rather
    than guess.

    ⚠ **Only escapes above U+007F are decoded.** The fault escapes the
    non-ASCII characters and nothing else, so an escape that spells an ASCII
    character was typed by the model and means those six characters -- leaving
    it alone is the difference between repairing the frame and editing its
    content. `\u0041` therefore survives a frame that is otherwise being
    repaired. (Surrogates are U+D800-DFFF, so the pair rejoining below is
    unaffected.)
    """

    def _decode(match: "re.Match[str]") -> str:
        codepoint = int(match.group(1), 16)
        return chr(codepoint) if codepoint > 127 else match.group(0)

    decoded = _ESCAPED_CODEPOINT.sub(_decode, value)
    if any("\ud800" <= character <= "\udfff" for character in decoded):
        try:
            decoded = decoded.encode("utf-16", "surrogatepass").decode("utf-16")
        except UnicodeError:
            return value
    return decoded


# --------- Tunables (abnormal ASR detection) ---------
LONG_WORD_SEC = 5.0
LONG_WORD_WORDS = 15
SPACE_DELIMITED_WORD_UNITS = 3.0
NON_SPACE_SCRIPT_CHAR_UNITS = 1.0
DIGIT_RUN_UNITS = 1.0
REPEAT_DETECT_MORE_THAN = 7
REPEAT_KEEP_RUN = 5
GROUP_REPEAT_MIN_COUNT = 4
GROUP_REPEAT_MIN_UNITS = 32.0
# Alignment-collapse word stacks (whisper-timestamped efficient path): words
# piled into a near-zero span show up as runs of frame-quantized ~20ms words.
# Calibrated on the collapse re-align experiment (out/collapse-exp, 2026-07-19):
# every confirmed real-line collapse had >=3 consecutive <=20ms words, while
# fast-but-correct speech only had isolated ones. 25ms tolerates rounding.
COLLAPSE_STACK_WORD_SEC = 0.025
COLLAPSE_STACK_MIN_RUN = 3
# Whisper's canonical Japanese end-credits hallucination. Shared by the
# stabilize phrase cleanup and the alignment-stage early exit for
# phrase-only collapse stacks.
COMMON_HALLUCINATION_TEXT = "ご視聴ありがとうございました"
# fw-refine disfluency marker word: a candidate span, not decoded content.
DISFLUENCY_WORD = "[*]"


def normalized_compact(text: str) -> str:
    """NFKC + casefold, punctuation and whitespace stripped.

    The shared comparison key for hallucination-phrase matching, ghost
    duplicate detection and second-model evidence."""

    return "".join(
        char.casefold()
        for char in unicodedata.normalize("NFKC", str(text))
        if not unicodedata.category(char).startswith("P") and not char.isspace()
    )
# Repeat detection never searches for a repeating pattern longer than this
# many characters; it bounds the O(text_len * motif_len) scans.
REPEAT_MATCH_MAX_CHARS = 100

# --------- Punctuation classification (split-boundary semantics) ---------
# Shared, direction-aware classification for subtitle split decisions: a cut
# between two words inspects the previous word's trailing character and the
# next word's leading character. Classes:
#   "sentence" — sentence-final (。．｡.！!？?…‥‼⁇⁈⁉): strongest break after.
#   "clause"   — clause-level pause (、，,､；;：:): weak break after.
#   "opening"  — opening bracket/quote (Ps/Pi: 「『（〔【《〈 " ( [ …):
#                glues RIGHTWARD — never break right after one.
#   "closing"  — closing bracket/quote (Pe/Pf: 」』）〕】》〉 " ) ] …):
#                glues LEFTWARD — never break right before one.
#   "other"    — remaining punctuation (・‐—/～ " ' etc.): mid-strength break.
#   "none"     — not punctuation.
SENTENCE_END_PUNCT = frozenset("。．｡.！!？?…‥‼⁇⁈⁉")
CLAUSE_PUNCT = frozenset("、，,､；;：:")


_opencc_t2s: Any = None


def t2s_converter() -> Any:
    """Return the shared opencc 繁→简 converter, or ``None`` if unavailable.

    ``opencc-python-reimplemented`` is a declared ``[harness]`` dependency, so a
    missing one means the environment drifted rather than an optional feature
    being switched off. Degrading silently is what let knowledge entries fork
    into 繁/简 duplicates unnoticed, so say it once per process.
    """

    global _opencc_t2s
    if _opencc_t2s is None:
        try:
            from opencc import OpenCC

            _opencc_t2s = OpenCC("t2s")
        except ImportError:
            _opencc_t2s = False
            current_reporter().warning(
                "opencc-unavailable",
                "opencc is unavailable, so 繁→简 normalization is off and "
                "traditional/simplified spellings will no longer match",
                action='install the harness extra (pip install -e ".[harness]")',
            )
    return _opencc_t2s or None


def punct_class(ch: str) -> str:
    """Classify one character for split-boundary decisions (see table above)."""

    if not ch:
        return "none"
    if ch in SENTENCE_END_PUNCT:
        return "sentence"
    if ch in CLAUSE_PUNCT:
        return "clause"
    category = unicodedata.category(ch)
    if category in ("Ps", "Pi"):
        return "opening"
    if category in ("Pe", "Pf"):
        return "closing"
    if category.startswith("P"):
        return "other"
    return "none"
INTERNAL_WORD_JOINERS = {"'", "’", "-", "·"}
NON_SPACE_DELIMITED_SCRIPT_MARKERS = (
    "CJK UNIFIED IDEOGRAPH",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "THAI",
    "LAO",
    "KHMER",
    "MYANMAR",
    "BOPOMOFO",
    "YI SYLLABLE",
    "YI RADICAL",
)
SPACE_DELIMITED_SCRIPT_MARKERS = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "ARABIC",
    "HEBREW",
    "DEVANAGARI",
    "BENGALI",
    "GURMUKHI",
    "GUJARATI",
    "ORIYA",
    "TAMIL",
    "TELUGU",
    "KANNADA",
    "MALAYALAM",
    "SINHALA",
    "GEORGIAN",
    "ARMENIAN",
    "ETHIOPIC",
)


def is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    return unicodedata.east_asian_width(ch) in {"W", "F"}


def _unicode_name(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return ""


def _is_non_space_script_letter(ch: str) -> bool:
    if not ch or not ch.isalpha():
        return False
    name = _unicode_name(ch)
    return any(marker in name for marker in NON_SPACE_DELIMITED_SCRIPT_MARKERS)


def _is_space_script_letter(ch: str) -> bool:
    if not ch or not ch.isalpha():
        return False
    if _is_non_space_script_letter(ch):
        return False
    name = _unicode_name(ch)
    return any(marker in name for marker in SPACE_DELIMITED_SCRIPT_MARKERS)


def _is_internal_joiner(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text) - 1:
        return False
    ch = text[index]
    if ch not in INTERNAL_WORD_JOINERS:
        return False
    prev = text[index - 1]
    nxt = text[index + 1]
    return (prev.isalpha() or prev.isdigit()) and (nxt.isalpha() or nxt.isdigit())


def _text_unit_symbols(text: str) -> List[Tuple[str, float]]:
    """Return normalized comparison symbols carrying the shared word-unit weights."""

    text = unicodedata.normalize("NFC", text or "")
    symbols: List[Tuple[str, float]] = []
    run_chars: List[str] = []
    run_has_space_script_letter = False
    run_has_digit = False

    def flush_space_delimited_run() -> None:
        nonlocal run_chars, run_has_space_script_letter, run_has_digit
        if run_has_space_script_letter:
            symbols.append(("".join(run_chars).casefold(), SPACE_DELIMITED_WORD_UNITS))
        elif run_has_digit:
            symbols.append(("".join(run_chars), DIGIT_RUN_UNITS))
        run_chars = []
        run_has_space_script_letter = False
        run_has_digit = False

    for index, ch in enumerate(text):
        if _is_space_script_letter(ch):
            run_chars.append(ch)
            run_has_space_script_letter = True
            continue
        if ch.isdigit():
            run_chars.append(ch)
            run_has_digit = True
            continue
        if unicodedata.category(ch).startswith("M"):
            if run_chars:
                run_chars.append(ch)
            continue
        if _is_internal_joiner(text, index):
            if run_chars:
                run_chars.append(ch)
            continue

        flush_space_delimited_run()
        if _is_non_space_script_letter(ch):
            symbols.append((ch.casefold(), NON_SPACE_SCRIPT_CHAR_UNITS))
        elif ch.isalpha():
            # Keep unclassified scripts conservative: one unit per letter.
            symbols.append((ch.casefold(), 1.0))

    flush_space_delimited_run()
    return symbols


def count_word_units(text: str) -> float:
    return sum(units for _symbol, units in _text_unit_symbols(text))


def detect_repeating_group_cycle(
    text: str,
    *,
    min_repeats: int = GROUP_REPEAT_MIN_COUNT,
    min_span_units: float = GROUP_REPEAT_MIN_UNITS,
) -> Optional[str]:
    """Describe the first exact local tandem repeat that crosses both thresholds."""

    unit_symbols = _text_unit_symbols(text)
    required_repeats = max(2, int(min_repeats))
    required_units = max(0.0, float(min_span_units))
    symbol_count = len(unit_symbols)
    if symbol_count < required_repeats:
        return None

    symbols = [symbol for symbol, _units in unit_symbols]
    unit_prefix = [0.0]
    for _symbol, units in unit_symbols:
        unit_prefix.append(unit_prefix[-1] + units)
    char_prefix = [0]
    for symbol in symbols:
        char_prefix.append(char_prefix[-1] + len(symbol))

    # Every symbol is at least one character, so motifs longer than the char
    # cap in symbols can be skipped outright.
    max_motif_len = min(symbol_count // required_repeats, REPEAT_MATCH_MAX_CHARS)
    for motif_len in range(1, max_motif_len + 1):
        max_start = symbol_count - required_repeats * motif_len
        for start in range(max_start + 1):
            motif_chars = char_prefix[start + motif_len] - char_prefix[start]
            if motif_chars > REPEAT_MATCH_MAX_CHARS:
                continue
            motif = symbols[start : start + motif_len]
            repeats = 1
            cursor = start + motif_len
            while (
                cursor + motif_len <= symbol_count
                and symbols[cursor : cursor + motif_len] == motif
            ):
                repeats += 1
                cursor += motif_len
            if repeats < required_repeats:
                continue
            span_units = unit_prefix[cursor] - unit_prefix[start]
            if span_units < required_units:
                continue
            motif_units = unit_prefix[start + motif_len] - unit_prefix[start]
            motif_text = " ".join(motif)
            return (
                "repeating_group_cycle "
                f"count={repeats} motif_units={motif_units:g} "
                f"span_units={span_units:g} motif='{motif_text[:40]}'"
            )
    return None


def coerce_optional_float(value: object) -> Optional[float]:
    """Return ``value`` as float, or None when missing or not coercible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def copy_float_fields(
    source: Dict[str, object],
    target: Dict[str, object],
    keys: Tuple[str, ...],
) -> None:
    """Copy coercible float fields from ``source`` onto ``target``."""

    for key in keys:
        value = coerce_optional_float(source.get(key))
        if value is not None:
            target[key] = value


def min_word_confidence(words: List[Dict[str, object]]) -> Optional[float]:
    """A synthetic word is only as reliable as its least-confident source."""

    confidences = [
        confidence
        for confidence in (
            coerce_optional_float(word.get("confidence")) for word in words
        )
        if confidence is not None
    ]
    return min(confidences) if confidences else None


def words_to_text(words: List[Dict[str, object]]) -> str:
    result_tokens: List[str] = []
    for word in words:
        token = word.get("word")
        if token is None:
            continue
        token = str(token)
        if not token:
            continue
        if result_tokens and bool(word.get("space_before", False)):
            result_tokens.append(" ")
        result_tokens.append(token)
    return "".join(result_tokens).strip()


def _char_repeat_key(ch: str) -> str:
    lowered = ch.lower()
    decomp = unicodedata.normalize("NFKD", lowered)
    base = "".join(c for c in decomp if not unicodedata.combining(c))
    return base or lowered


def _collapse_exact_repeating_unit(
    word: str,
    *,
    detect_more_than: int,
    keep_repeats: int,
) -> Tuple[str, bool]:
    n = len(word)
    if n <= 1:
        return word, False
    min_repeat_count = max(0, int(detect_more_than)) + 1
    max_unit_len = min(n // min_repeat_count, REPEAT_MATCH_MAX_CHARS)
    for unit_len in range(1, max_unit_len + 1):
        if n % unit_len != 0:
            continue
        repeat = n // unit_len
        if repeat <= detect_more_than:
            continue
        unit = word[:unit_len]
        if unit * repeat == word:
            return unit * max(1, int(keep_repeats)), True
    return word, False


def _collapse_repeating_char_runs(
    word: str,
    *,
    detect_more_than: int,
    keep_repeats: int,
) -> Tuple[str, bool]:
    if not word:
        return word, False
    detect_threshold = max(0, int(detect_more_than))
    keep_count = max(1, int(keep_repeats))
    chars = list(word)
    keys = [_char_repeat_key(ch) for ch in chars]
    out: List[str] = []
    i = 0
    changed = False
    while i < len(chars):
        j = i + 1
        while j < len(chars) and keys[j] == keys[i]:
            j += 1
        run_len = j - i
        if run_len > detect_threshold:
            out.extend(chars[i : i + keep_count])
            changed = True
        else:
            out.extend(chars[i:j])
        i = j
    collapsed = "".join(out)
    return collapsed, changed


def collapse_repeating_pattern(
    word: str,
    *,
    detect_more_than: int = REPEAT_DETECT_MORE_THAN,
    keep_repeats: int = REPEAT_KEEP_RUN,
) -> Tuple[str, bool]:
    collapsed, changed = _collapse_exact_repeating_unit(
        word,
        detect_more_than=detect_more_than,
        keep_repeats=keep_repeats,
    )
    if changed:
        return collapsed, True
    return _collapse_repeating_char_runs(
        word,
        detect_more_than=detect_more_than,
        keep_repeats=keep_repeats,
    )


def _segment_text_and_word_spans(
    words: List[Dict[str, object]],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Build the exact segment text and the character span owned by each word."""

    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    has_text = False
    for word in words:
        token = str(word.get("word") or "")
        span_start = cursor
        if token and has_text and bool(word.get("space_before", False)):
            parts.append(" ")
            cursor += 1
        parts.append(token)
        cursor += len(token)
        spans.append((span_start, cursor))
        has_text = has_text or bool(token)
    return "".join(parts), spans


def _find_exact_repeating_text_span(
    text: str,
    *,
    detect_more_than: int,
    keep_repeats: int,
) -> Optional[Tuple[int, int, str, int]]:
    """Find the most removable local exact tandem repeat in raw segment text.

    Unlike the abnormal-result detector, this cleanup deliberately compares raw
    characters. Punctuation and spaces therefore participate in the motif.
    """

    min_repeat_count = max(0, int(detect_more_than)) + 1
    keep_count = max(1, int(keep_repeats))
    text_len = len(text)
    if text_len < min_repeat_count:
        return None

    best: Optional[Tuple[int, int, str, int]] = None
    best_key: Optional[Tuple[int, int, int, int]] = None
    for start in range(text_len):
        max_motif_len = min(
            (text_len - start) // min_repeat_count,
            REPEAT_MATCH_MAX_CHARS,
        )
        for motif_len in range(1, max_motif_len + 1):
            motif = text[start : start + motif_len]
            repeats = 1
            cursor = start + motif_len
            while (
                cursor + motif_len <= text_len
                and text[cursor : cursor + motif_len] == motif
            ):
                repeats += 1
                cursor += motif_len
            if repeats < min_repeat_count:
                continue
            removed_chars = max(0, repeats - keep_count) * motif_len
            if removed_chars <= 0:
                continue
            # Prefer the largest actual reduction, then more repetitions, then
            # the earliest location and the shortest/fundamental motif.
            key = (removed_chars, repeats, -start, -motif_len)
            if best_key is None or key > best_key:
                best = (start, cursor, motif, repeats)
                best_key = key
    return best


def collapse_repeating_segment_words(
    words: List[Dict[str, object]],
    *,
    detect_more_than: int = REPEAT_DETECT_MORE_THAN,
    keep_repeats: int = REPEAT_KEEP_RUN,
    warn_context: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Recursively collapse exact local repeats over a complete ASR segment.

    Every word touched by a collapsed text range is replaced by one synthetic
    word spanning all of their timestamps. Rebuilding the segment after each
    reduction lets independent or newly exposed repeat regions be handled too.
    """

    collapsed_words = [dict(word) for word in words]
    keep_count = max(1, int(keep_repeats))
    while collapsed_words:
        text, spans = _segment_text_and_word_spans(collapsed_words)
        repeat = _find_exact_repeating_text_span(
            text,
            detect_more_than=detect_more_than,
            keep_repeats=keep_count,
        )
        if repeat is None:
            break

        repeat_start, repeat_end, motif, repeat_count = repeat
        affected = [
            index
            for index, (word_start, word_end) in enumerate(spans)
            if word_start < repeat_end and word_end > repeat_start
        ]
        if not affected:
            break
        first_index = affected[0]
        last_index = affected[-1]
        merge_start = spans[first_index][0]
        merge_end = spans[last_index][1]
        replacement = motif * keep_count
        merged_text = (
            text[merge_start:repeat_start]
            + replacement
            + text[repeat_end:merge_end]
        ).strip()
        source_words = collapsed_words[first_index : last_index + 1]
        merged_word: Dict[str, object] = {
            "start": float(source_words[0]["start"]),
            "end": float(source_words[-1]["end"]),
            "word": merged_text,
            "space_before": bool(source_words[0].get("space_before", False)),
        }
        confidence = min_word_confidence(source_words)
        if confidence is not None:
            merged_word["confidence"] = confidence
        if warn_context:
            # Per occurrence, and a run can have many: this is the cleanup
            # working, not something a person has to act on.
            current_reporter().debug(
                "merged repeating segment text "
                f"(count={repeat_count} -> keep={keep_count}, "
                f"words={len(source_words)}) in {warn_context}, "
                f"motif='{motif[:40]}'"
            )
        collapsed_words = (
            collapsed_words[:first_index]
            + [merged_word]
            + collapsed_words[last_index + 1 :]
        )
    return collapsed_words


def collapse_stack_run(
    words: List[Dict[str, object]],
    *,
    stack_word_sec: float = COLLAPSE_STACK_WORD_SEC,
    min_run: int = COLLAPSE_STACK_MIN_RUN,
) -> Optional[Tuple[int, int]]:
    """(start_index, run_length) of the longest alignment-collapse stack —
    >=min_run consecutive words whose duration is <=stack_word_sec — or None.
    Words must already be ordered by time."""

    run_start: Optional[int] = None
    best: Optional[Tuple[int, int]] = None  # (run_len, start_index)
    for i, w in enumerate(words):
        duration = float(w.get("end", 0.0)) - float(w.get("start", 0.0))
        if duration <= stack_word_sec and str(w.get("word") or "").strip():
            if run_start is None:
                run_start = i
            run_len = i - run_start + 1
            if best is None or run_len > best[0]:
                best = (run_len, run_start)
        else:
            run_start = None
    if best is None or best[0] < min_run:
        return None
    run_len, start_index = best
    return start_index, run_len


def detect_collapse_word_stack(
    words: List[Dict[str, object]],
    *,
    stack_word_sec: float = COLLAPSE_STACK_WORD_SEC,
    min_run: int = COLLAPSE_STACK_MIN_RUN,
) -> Optional[str]:
    """Issue string for the collapse stack in ``words``, or None."""

    run = collapse_stack_run(
        words, stack_word_sec=stack_word_sec, min_run=min_run
    )
    if run is None:
        return None
    start_index, run_len = run
    stacked = words[start_index : start_index + run_len]
    text = "".join(str(w.get("word") or "") for w in stacked)
    return (
        f"collapse_word_stack count={run_len} "
        f"span={float(stacked[0]['start']):.3f}-{float(stacked[-1]['end']):.3f} "
        f"token='{text[:40]}'"
    )


def detect_abnormal_asr_words(
    per_segment: List[List[Dict[str, object]]],
    *,
    long_word_sec: float = LONG_WORD_SEC,
    long_word_words: int = LONG_WORD_WORDS,
    repeat_detect_more_than: int = REPEAT_DETECT_MORE_THAN,
    repeat_keep_run: int = REPEAT_KEEP_RUN,
    group_repeat_min_count: int = GROUP_REPEAT_MIN_COUNT,
    group_repeat_min_units: float = GROUP_REPEAT_MIN_UNITS,
    collapse_stack_word_sec: float = COLLAPSE_STACK_WORD_SEC,
    collapse_stack_min_run: int = COLLAPSE_STACK_MIN_RUN,
) -> List[str]:
    issues: List[str] = []
    for words in per_segment:
        # ``[*]`` marks an attention-detected disfluency span, not decoded
        # content: its (often long) span is resolved by the word-start rules
        # downstream and must not read as a stretched/repeated word here.
        words = [
            w for w in words if str(w.get("word") or "") != DISFLUENCY_WORD
        ]
        if not words:
            continue
        ordered = sorted(words, key=lambda x: (x["start"], x["end"]))
        stack_issue = detect_collapse_word_stack(
            ordered,
            stack_word_sec=collapse_stack_word_sec,
            min_run=collapse_stack_min_run,
        )
        if stack_issue:
            issues.append(stack_issue)
        run_count = 1
        prev_word = str(ordered[0].get("word") or "")
        for i, w in enumerate(ordered):
            w_text = str(w.get("word") or "")
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", 0.0))

            if (w_end - w_start) >= long_word_sec:
                issues.append(
                    f"long_word_duration={w_end - w_start:.2f}s token='{w_text[:40]}'"
                )
            word_units = count_word_units(w_text)
            if word_units >= long_word_words:
                issues.append(
                    f"long_word_token=units({word_units:g}) token='{w_text[:40]}'"
                )

            _collapsed_text, collapsed = collapse_repeating_pattern(
                w_text,
                detect_more_than=repeat_detect_more_than,
                keep_repeats=repeat_keep_run,
            )
            if collapsed:
                issues.append(f"repeating_token token='{w_text[:40]}'")

            if i == 0:
                continue
            if w_text == prev_word and w_text:
                run_count += 1
                if run_count > repeat_detect_more_than:
                    issues.append(
                        f"repeating_word_run count={run_count} token='{w_text[:40]}'"
                    )
            else:
                run_count = 1
                prev_word = w_text

    group_words = [word for words in per_segment for word in words]
    group_words.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    group_cycle = detect_repeating_group_cycle(
        words_to_text(group_words),
        min_repeats=group_repeat_min_count,
        min_span_units=group_repeat_min_units,
    )
    if group_cycle:
        issues.append(group_cycle)
    return issues


def cleanup_asr_words_for_fallback(
    words: List[Dict[str, object]],
    *,
    segment_start: float,
    segment_end: float,
    long_word_sec: float = LONG_WORD_SEC,
    long_word_words: int = LONG_WORD_WORDS,
    repeat_detect_more_than: int = REPEAT_DETECT_MORE_THAN,
    repeat_keep_run: int = REPEAT_KEEP_RUN,
) -> List[Dict[str, object]]:
    ordered = sorted(words, key=lambda x: (x["start"], x["end"]))
    for w in ordered:
        w_text = str(w.get("word") or "")
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        if (w_end - w_start) >= long_word_sec:
            # One per suspect word: a signal for whoever is investigating the
            # timeline, not a queue of warnings for the person who wanted
            # subtitles.
            current_reporter().debug(
                "long word duration "
                f"{w_end - w_start:.2f}s in segment "
                f"{segment_start}-{segment_end}, word='{w_text}'"
            )

    filtered = collapse_repeating_segment_words(
        ordered,
        detect_more_than=repeat_detect_more_than,
        keep_repeats=repeat_keep_run,
        warn_context=f"segment {segment_start}-{segment_end}",
    )
    for w in filtered:
        w_text = str(w.get("word") or "")
        word_units = count_word_units(w_text)
        if word_units >= long_word_words:
            current_reporter().debug(
                "long word token "
                f"(units={word_units:g}) in segment "
                f"{segment_start}-{segment_end}, word='{w_text}'"
            )
    return filtered
