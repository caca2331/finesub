"""How the recall repair (`recall_repair`) compares two transcriptions of one
stretch of speech: by sound, so a word one model writes in kanji and the
other in kana, or a traditional and a simplified hanzi, still match.

Two readings, kept apart because the accepted rules were measured with each
(docs/asr-stabilize.md, "补救：坍缩行与空档"):

* `content_sounds` -- the gap fill's gate: the text normalised (`norm`),
  interjections and backchannels taken off its ends (`content`), read as a
  whole (`reading`: hiragana for Japanese, toneless pinyin syllables for
  Mandarin, letters otherwise), compared by longest common subsequence.
* `sounds` / `pairs` -- the collapse repair's coverage: every normalised
  character read on its own, numerals as written and fillers kept, matched
  by contiguous blocks of at least BLOCK sounds.
* `units` / `unit_pairs` -- what the collapse ladder and the in-row fill
  align: the sounds in a language written without spaces (UNSPACED), matched
  by blocks of BLOCK; the words elsewhere, matched by blocks of WORD_BLOCK. A
  letter is too loose a unit: two English texts share some pair of letters
  almost anywhere; one shared word is seldom chance.

`norm`: NFKC, lower case, punctuation and spaces removed, katakana folded to
hiragana, traditional Chinese folded to simplified (Mandarin and Cantonese), runs of two or more kanji
numerals turned into digits (a single kanji digit stays: 一緒, 二人 are words).
The readings depend on the versions of pykakasi, pypinyin and opencc, which
are pinned for that reason.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

#: Content characters a stretch needs to count as speech of its own.
SUBST = 3
#: Contiguous sounds two texts must share for `pairs` to match them: one
#: shared kana or syllable is as often chance as the same speech.
BLOCK = 2
#: Languages written without spaces between words (as Whisper splits words,
#: `recognition/fw_refine.should_use_space`): `units` are their sounds.
UNSPACED = ("ja", "zh", "yue", "th", "lo", "my")
#: `unit_pairs`'s block where units are words.
WORD_BLOCK = 1

_EDGE = {
    "ja": ("そうですね|ですね|うんうん|うん|ううん|はいはい|はい|ええ|えー|えっ|ああ|あー|あっ|おお|おー|おっ|へえ|へー|ほう|ほー|"
           "そうそう|そうだね|そうね|そうなんだ|そうなの|そう|なるほど|ねえ|ねー|んー|ふふ|はは|わあ|わー|うわ|"
           "うーん|ふーん|はーん|はあ|はー|はっ|ふう|ふー|さあ|まあ|ほい|へい|えっと|ええと|えーと|あのー|うふふ|あはは"),
    "zh": "对对对|对对|是的|是吧|对吧|对啊|是啊|好的|好吧|嗯嗯|哈哈|哎呀|哎哟|哦哦|嗯哼",
    "en": "yeah|yes|oh|uh|um|mm|hmm|okay|ok|yep|mhm|uhhuh",
}
# Single characters count as filler only when the whole run is made of fillers, never stripped from a longer run's ends
_SINGLE = {"ja": "え|あ|お|ね|ん|ふ|は|よ|な|わ", "zh": "嗯|啊|哦|噢|呃|额|哈|对|是|好|行|嘛|呢|吧|哎|唉|诶|欸", "en": "ah|right"}
# Other languages have no list: nothing is taken off as filler there.


@lru_cache(maxsize=None)
def _only_fillers(lang: str) -> re.Pattern:
    return re.compile(f"(?:{_EDGE[lang]}|{_SINGLE[lang]})+")


@lru_cache(maxsize=1)
def _t2s():
    import opencc
    return opencc.OpenCC("t2s")


@lru_cache(maxsize=1)
def _kakasi():
    import pykakasi
    return pykakasi.kakasi()


def readings_available() -> bool:
    """Whether the three reading libraries import."""
    try:
        import opencc  # noqa: F401
        import pykakasi  # noqa: F401
        import pypinyin  # noqa: F401
    except ImportError:
        return False
    return True


_DROP = re.compile(r"[\W_]+", re.UNICODE)


def _kata2hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10 ** 4, "億": 10 ** 8, "亿": 10 ** 8}
_NUMERAL = re.compile(f"[{''.join(_DIGIT)}{''.join(_UNIT)}{''.join(_BIG)}]{{2,}}")


def _numeral(run: str) -> str:
    if all(c in _DIGIT for c in run):  # read digit by digit: 二〇二四
        return "".join(str(_DIGIT[c]) for c in run)
    total = section = 0
    cur = None
    for c in run:
        if c in _DIGIT:
            cur = _DIGIT[c]
        elif c in _UNIT:
            section += (1 if cur is None else cur) * _UNIT[c]
            cur = None
        else:
            total += ((section + (cur or 0)) or 1) * _BIG[c]
            section, cur = 0, None
    return str(total + section + (cur or 0))


_COMBINING_KANA = "゙゚ﾞﾟ"


def norm_map(text: str, lang: str) -> list[tuple[str, int]]:
    """Normalise `text` (numerals aside) and give, for every output character,
    the index of the input character it came from. A base character and the
    combining marks after it (half-width or decomposed dakuten) are
    normalised together."""
    text = text or ""
    out: list[tuple[str, int]] = []
    k = 0
    while k < len(text):
        end = k + 1
        while end < len(text) and (unicodedata.combining(text[end]) or text[end] in _COMBINING_KANA):
            end += 1
        piece = _DROP.sub("", unicodedata.normalize("NFKC", text[k:end]).lower())
        if lang in ("zh", "yue") and piece:
            piece = _t2s().convert(piece)
        out += [(_kata2hira(x), k) for x in piece]
        k = end
    return out


def norm(text: str, lang: str) -> str:
    chars = "".join(c for c, _ in norm_map(text, lang))
    if lang not in ("ja", "zh"):
        return chars
    out, last = [], 0
    for m in _NUMERAL.finditer(chars):
        out += [chars[last:m.start()], _numeral(m.group())]
        last = m.end()
    return "".join(out) + chars[last:]


@lru_cache(maxsize=65536)
def reading(text: str, lang: str) -> str:
    if not text:
        return ""
    if lang == "ja":
        return "".join(item["hira"] for item in _kakasi().convert(text))
    if lang == "zh":
        from pypinyin import lazy_pinyin
        return " ".join(lazy_pinyin(text))
    return text


def content(text: str, lang: str) -> str:
    """`text` (already normalised) without the interjections at its ends:
    empty when it is nothing but fillers. Multi-character fillers are
    stripped from the ends, single characters never (`僕はね` keeps its `は`).
    A language without a filler list keeps all of it."""
    if lang not in _EDGE:
        return text or ""
    if not text or _only_fillers(lang).fullmatch(text):
        return ""
    edge = _EDGE[lang]
    head = re.match(f"(?:{edge})+", text)
    a = head.end() if head else 0
    tail = re.search(f"(?:{edge})+$", text[a:])
    b = a + tail.start() if tail else len(text)
    return text[a:b]


def content_words(text: str, lang: str) -> str:
    """`content` by `units` where they are words: empty when `text` is
    nothing but fillers, else its words with the multi-character fillers at
    either end taken off, joined (a filler's letters inside a word, "look",
    "eyes", stay; "right" stays at an end, as in `content`). In an UNSPACED
    language, `content` of `norm`."""
    if lang in UNSPACED:
        return content(norm(text or "", lang), lang)
    ws = [u for u, _, _ in units(text, lang)]
    if lang not in _EDGE:
        return "".join(ws)
    if _only_fillers(lang).fullmatch("".join(ws)):
        return ""
    edge = re.compile(f"(?:{_EDGE[lang]})+").fullmatch
    while ws and edge(ws[0]):
        ws.pop(0)
    while ws and edge(ws[-1]):
        ws.pop()
    return "".join(ws)


def substantive(text: str, lang: str) -> bool:
    """`text` (already normalised) holds at least SUBST content characters."""
    return len(content(text, lang)) >= SUBST


def content_sounds(text: str, lang: str) -> list[str]:
    """The reading of `text`'s content (normalised, fillers off) as tokens."""
    r = reading(content(norm(text or "", lang), lang), lang)
    return r.split() if lang == "zh" else list(r)


def _lcs_table(a, b) -> list[list[int]]:
    """LCS lengths of every suffix pair."""
    L = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if a[i] == b[j] else max(L[i + 1][j], L[i][j + 1])
    return L


def lcs(a, b) -> int:
    return _lcs_table(a, b)[0][0] if a and b else 0


def unmatched(a, b) -> list:
    """The items of `a` a longest common subsequence with `b` leaves out, in order."""
    L, out, i, j = _lcs_table(a, b), [], 0, 0
    while i < len(a):
        if j < len(b) and a[i] == b[j]:
            i, j = i + 1, j + 1
        elif j < len(b) and L[i][j + 1] > L[i + 1][j]:
            j += 1
        else:
            out.append(a[i])
            i += 1
    return out


def missing(row: list[str], own: list[str]) -> list[str]:
    """The part of a row's reading its own span's decode lacks, as one piece:
    all of it when the decode holds under a quarter of it (the row's speech is
    elsewhere), else the longer unmatched run at its start or end (misplaced
    speech leaves a row whole or by one edge; tokens unmatched here and there
    are a decode's differences)."""
    L, keep, i, j = _lcs_table(row, own), [], 0, 0
    while i < len(row) and j < len(own):
        if row[i] == own[j]:
            keep.append(i)
            i, j = i + 1, j + 1
        elif L[i][j + 1] > L[i + 1][j]:
            j += 1
        else:
            i += 1
    if len(keep) < 0.25 * len(row) or not keep:
        return row
    head, tail = row[:keep[0]], row[keep[-1] + 1:]
    return head if len(head) > len(tail) else tail


@lru_cache(maxsize=65536)
def _char_sound(ch: str, lang: str) -> str:
    if lang == "zh":
        from pypinyin import lazy_pinyin
        return " ".join(lazy_pinyin(ch))
    return ch


def sounds(text: str, lang: str) -> list[tuple[str, int]]:
    """The sounds of `text`, normalised, each with the index in `text` of the character it came from."""
    chars = norm_map(text or "", lang)
    if not chars:
        return []
    if lang == "ja":
        out, k = [], 0
        for item in _kakasi().convert("".join(c for c, _ in chars)):
            n, hira = len(item["orig"]), item["hira"]
            out += [(h, chars[min(k + i * n // max(len(hira), 1), len(chars) - 1)][1]) for i, h in enumerate(hira)]
            k += n
        return out
    if lang == "zh":
        return [(s, i) for c, i in chars for s in _char_sound(c, lang).split()]
    return [(c, i) for c, i in chars]


def pairs(a: list[str], b: list[str], block: int = BLOCK) -> list[tuple[int, int]]:
    """Index pairs of `a` and `b` matched by contiguous blocks (the longest
    common block, then either side of it) of at least `block` items."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return [(m.a + k, m.b + k) for m in sm.get_matching_blocks() if m.size >= block for k in range(m.size)]


def unit_pairs(a: list[str], b: list[str], lang: str) -> list[tuple[int, int]]:
    """`pairs` of two `units` lists: blocks of BLOCK sounds, or WORD_BLOCK words."""
    return pairs(a, b, BLOCK if lang in UNSPACED else WORD_BLOCK)


def units(text: str, lang: str) -> list[tuple[str, int, int]]:
    """`text` as units, each with the characters of `text` it spans ([start,
    end)): its sounds (`sounds`) in an UNSPACED language, each spanning the
    character it came from; elsewhere its words (runs of letters, marks,
    digits and apostrophes, normalised), each spanning its first to last
    character that normalising keeps. A span also takes the marks that follow
    it (Thai and Devanagari vowel and tone signs, a decomposed dakuten), so a
    cut or an insertion never parts a letter from its own signs; normalising
    drops them, so matching reads those scripts without their signs."""
    text = text or ""
    if lang in UNSPACED:
        return [(s, i, _past_marks(text, i + 1)) for s, i in sounds(text, lang)]
    out = []
    for a, b in _word_spans(text):
        chars = norm_map(text[a:b], lang)
        if chars:
            out.append(("".join(c for c, _ in chars), a + chars[0][1], _past_marks(text, a + chars[-1][1] + 1)))
    return out


def _past_marks(text: str, end: int) -> int:
    while end < len(text) and unicodedata.category(text[end])[0] == "M":
        end += 1
    return end


def _word_spans(text: str) -> list[tuple[int, int]]:
    """[start, end) of each run of letters, marks, digits and apostrophes."""
    out, a = [], None
    for k, c in enumerate(text + " "):
        if c in "'’" or unicodedata.category(c)[0] in "LMN":
            a = k if a is None else a
        elif a is not None:
            out.append((a, k))
            a = None
    return out


def _share(r: list[str], q: list[str], matched) -> float:
    return len({j for _, j in matched(q, r)}) / len(r) if r and q else 0.0


def cover(text: str, heard: str, lang: str, head: bool = False) -> float:
    """The share of `text`'s sounds (its first half with `head`) that `heard`'s sounds match by contiguous blocks."""
    r = [s for s, _ in sounds(text, lang)]
    return _share(r[:max(1, len(r) // 2)] if head else r, [s for s, _ in sounds(heard or "", lang)], pairs)


def unit_cover(text: str, heard: str, lang: str) -> float:
    """`cover` by `units` and `unit_pairs`: the same as `cover` in an UNSPACED language."""
    return _share([u for u, _, _ in units(text, lang)], [u for u, _, _ in units(heard or "", lang)],
                  lambda a, b: unit_pairs(a, b, lang))
