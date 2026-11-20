"""Which game-glossary names an ASR row carries: written verbatim, or by sound (Japanese only).

**Verbatim** (``verbatim``): a name form of at least 2 characters inside the row, longer forms masking the forms
inside them (リンネア hides リン); a short all-katakana form (at most ``SHORT_KATAKANA_MAX``) must not sit inside a
longer katakana run (クレー in クレープ), though one trailing ``ー`` is allowed. Pure Python.

**By sound** (``SoundMatcher``): a misheard name is a word or a run of words in the row, so terms are compared with
whole spans of the row (``spans``), each turned into phonemes (consonant + vowel per mora; ー, ッ, ン are phonemes of
their own). A term's cost against a span is a weighted edit distance over phonemes divided by the term's phoneme
count: sounds ASR confuses (voicing, n/r/d, i/u ...) are cheap, others dear, a dropped ー or ッ nearly free, the
term's tail may be cut (ASR truncates long names) while ``KEEP`` of it stays, and the term's first consonant heard
as h costs ``near`` (ヘイワット for テイワット). A span kind passes at its own limit (``LIMITS``); a katakana term on a
kanji word span at ``KANA_ON_WORD``; hiragana runs only stand in for katakana terms (さんご is coral, not 珊瑚). Terms
the caller marks *frequent* (written verbatim elsewhere in the recording) pass katakana spans at ``FREQUENT``.
A (term, span surface) pair whose rate on the glossary's background rows reaches ``NOISE_LINE`` is dropped.
The thresholds were chosen on held-out recordings (docs/llm_design_notes.md, the game glossary's decision record).

Needs the ``kb-phonetic`` extra (pyopenjtalk-plus, which brings Sudachi and its core dictionary, and rapidfuzz).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

# --- verbatim -----------------------------------------------------------------------

SHORT_KATAKANA_MAX = 3
_KATAKANA_CH = re.compile(r"[ァ-ヺー]")


def _verbatim_at(text: str, form: str) -> list[int]:
    """Where ``form`` is written in ``text`` as a whole name: a short katakana name not inside a longer katakana word
    (ゼン in ゼンマイ), a Latin one not inside a longer word (Ei in Either)."""

    starts, i = [], text.find(form)
    edge = len(form) <= SHORT_KATAKANA_MAX and all(_KATAKANA_CH.fullmatch(ch) for ch in form)
    while i >= 0:
        j = i + len(form)
        if j < len(text) and text[j] == "ー":
            j += 1
        if not edge or not ((i and _KATAKANA_CH.fullmatch(text[i - 1])) or
                            (j < len(text) and _KATAKANA_CH.fullmatch(text[j]))):
            if not ((_latin(form[0]) and i and _latin(text[i - 1])) or
                    (_latin(form[-1]) and j < len(text) and _latin(text[j]))):
                starts.append(i)
        i = text.find(form, i + 1)
    return starts


def _latin(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def verbatim(text: str, forms: Sequence[str]) -> set[str]:
    """The ``forms`` written in ``text``; ``forms`` must be sorted longest first (``by_length``)."""

    found: set[str] = set()
    for form in forms:
        if len(form) >= 2 and form in text and (starts := _verbatim_at(text, form)):
            found.add(form)
            for i in reversed(starts):
                text = text[:i] + "\0" + text[i + len(form):]
    return found


def by_length(forms: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(forms), key=lambda f: (-len(f), f)))


# --- phonemes -----------------------------------------------------------------------

_ROWS = {  # consonant -> kana for vowels a i u e o
    "": "アイウエオ", "k": "カキクケコ", "g": "ガギグゲゴ", "s": "サシスセソ", "z": "ザジズゼゾ",
    "t": "タチツテト", "d": "ダヂヅデド", "n": "ナニヌネノ", "h": "ハヒフヘホ", "b": "バビブベボ",
    "p": "パピプペポ", "m": "マミムメモ", "y": "ヤ_ユ_ヨ", "r": "ラリルレロ", "w": "ワ___ヲ",
}
_CV = {ch: (c, v) for c, row in _ROWS.items() for v, ch in enumerate(row) if ch != "_"}
_SMALL = "ァィゥェォャュョ"
_SMALL_V = dict(zip("ァィゥェォ", "aiueo"))
_SMALL_Y = dict(zip("ャュョ", "auo"))
VOWELS = set("aiueo")


def _kata(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def _units(kana: str) -> list[str]:
    """Kana grouped into morae: a small kana joins the one before it (ティ, シャ)."""

    out: list[str] = []
    for ch in kana:
        if out and ch in _SMALL:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def phonemes(kana: str, reading: bool = False) -> tuple[str, ...]:
    """``reading``: the kana is a reading (glossary, pyopenjtalk, Sudachi), where えい / おう / a doubled vowel are a
    long vowel (しょうり, ケイヤク = ショーリ, ケーヤク); a katakana spelling keeps them (テイワット is te-i)."""

    out: list[str] = []
    for unit in _units(_kata(re.sub(r"[^ァ-ヶーぁ-ゖ]", "", kana))):
        head, rest = unit[0], unit[1:]
        if head in "ーッン":
            out.append({"ー": ":", "ッ": "Q", "ン": "N"}[head])
            continue
        if head == "ヴ":
            c, v = "b", 2
        elif head in _CV:
            c, v = _CV[head]
        elif head in _SMALL:  # a stray small vowel: its own mora
            c, v = "", ("aiueo".index(_SMALL_V[head]) if head in _SMALL_V else 0)
        else:
            continue
        vowel = "aiueo"[v]
        if rest and rest[0] in _SMALL_Y:
            c, vowel = c + "y", _SMALL_Y[rest[0]]
        elif rest and rest[0] in _SMALL_V:
            vowel = _SMALL_V[rest[0]]
        if c:
            out.append(c)
        prev = out[-1] if out else ""
        long = reading and not c and (prev == vowel or (prev, vowel) in (("e", "i"), ("o", "u")))
        out.append(":" if long else vowel)
    return tuple(out)


def morae(ph: tuple[str, ...]) -> int:
    return sum(p in VOWELS or p in ":QN" for p in ph)


# --- cost ---------------------------------------------------------------------------

MIN_MORA = 3       # terms and spans shorter than this are left out (これ, こと match too many short names)
KEEP = 0.6         # a truncated span keeps at least this share of the term's phonemes ...
KEEP_MORA = 4      # ... and at least this many morae
COSTS = {"voicing": 0.25, "near": 0.4, "far": 0.9, "indel": 1.0, "light_indel": 0.4, "free_indel": 0.1,
         "trunc": 0.15}
_NEAR_V = {frozenset(p) for p in ("iu", "ei", "ou", "oa", "ue")}
_VOICING = {frozenset(p) for p in (("k", "g"), ("s", "z"), ("t", "d"), ("h", "b"), ("h", "p"), ("p", "b"))}
_NEAR_C = {frozenset(p) for p in (("n", "r"), ("r", "d"), ("n", "d"), ("m", "b"), ("m", "n"), ("g", "d"), ("t", "p"),
                                  ("w", "b"), ("k", "t"), ("s", "h"), ("ty", "sy"))}
_LIGHT = {"i", "u", "N", "r", "y", "h", "w"}


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if {a, b} <= VOWELS | {":"} and ":" in (a, b):  # ー heard as イ / ウ (ファデュイ -> ファリュー)
        return COSTS["near"]
    if (a in VOWELS) != (b in VOWELS) or a in ":QN" or b in ":QN":
        return 9.0
    pair = frozenset((a, b))
    if a in VOWELS:
        return COSTS["near"] if pair in _NEAR_V else COSTS["far"]
    if pair in _VOICING:
        return COSTS["voicing"]
    ba, bb = a.rstrip("y"), b.rstrip("y")
    if ba == bb:
        return COSTS["voicing"]
    return COSTS["near"] if pair in _NEAR_C or frozenset((ba, bb)) in _NEAR_C else COSTS["far"]


def _indel_cost(p: str) -> float:
    if p in (":", "Q"):
        return COSTS["free_indel"]
    return COSTS["light_indel"] if p in _LIGHT else COSTS["indel"]


@lru_cache(maxsize=1 << 18)
def cost(term: tuple[str, ...], span: tuple[str, ...]) -> float:
    """Weighted edit cost of the whole term against the whole span, per term phoneme; the term's tail may be cut,
    and its first consonant heard as h (or hy) costs ``near`` (any first consonant at that price, or a dropped one,
    matched loanwords too often: ページ for セージ)."""

    n, m = len(term), len(span)
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + _indel_cost(span[j - 1])
    best = math.inf
    keep = n
    if morae(term) >= KEEP_MORA + 1:
        keep = max(math.ceil(KEEP * n), next(i for i in range(n + 1) if morae(term[:i]) >= KEEP_MORA))
    for i in range(1, n + 1):
        ti = term[i - 1]
        onset = i == 1 and ti not in VOWELS
        cur = [prev[0] + _indel_cost(ti)] + [0.0] * m
        for j in range(1, m + 1):
            sub = _sub_cost(ti, span[j - 1])
            if onset and span[j - 1] in ("h", "hy"):
                sub = min(sub, COSTS["near"])
            cur[j] = min(prev[j - 1] + sub, prev[j] + _indel_cost(ti), cur[j - 1] + _indel_cost(span[j - 1]))
        if keep <= i < n:
            best = min(best, cur[m] + COSTS["trunc"] * (n - i))
        prev = cur
    return min(best, prev[m]) / n


# --- spans --------------------------------------------------------------------------

MAX_WORDS = 3
_SKIP_POS = {"助詞", "助動詞", "記号", "フィラー", "感動詞"}
_KATA_RUN = re.compile(r"[ァ-ヺー・･]+")
_HIRA_RUN = re.compile(r"[ぁ-ゖー]+")
_KATA = re.compile(r"[ァ-ヺ]")
_KANJI = re.compile(r"[一-鿿々]")


def spans(text: str) -> list[tuple[str, tuple[str, ...], str]]:
    """(surface, phonemes, kind) of the row's candidate spans.

    ``kata``: every katakana run; ``part``: its prefixes and suffixes at mora boundaries (a name joined to another
    word: チャスカリネ); the spelling is the reading (pyopenjtalk cuts unknown katakana words mid-word).
    ``mix``: a katakana run and the 1-2 words with kanji right after it (ライデン将軍).
    ``word``: 1-3 pyopenjtalk words with no katakana, not starting or ending on a function word, read by pyopenjtalk
    and by Sudachi, which reads coined compounds by their on-readings (粘素 ネンソ for 燃素); ``hira`` when they have
    no kanji either. ``hrun``: a whole hiragana run read as written (pyopenjtalk cuts おでっとだ into pieces);
    ``hpre``: its prefixes at mora boundaries (the name before a particle: おでっと).
    """

    import pyopenjtalk

    out = []
    for run in _KATA_RUN.finditer(text):
        u = _units(re.sub(r"[・･]", "", run.group()))
        whole = "".join(u)
        pieces = {"".join(u[:k]) for k in range(1, len(u))} | {"".join(u[k:]) for k in range(1, len(u))}
        for piece, kind in [(whole, "kata"), *((x, "part") for x in pieces - {whole})]:
            ph = phonemes(piece)
            if morae(ph) >= MIN_MORA and not piece.startswith("ー"):
                out.append((piece, ph, kind))
    for run in _HIRA_RUN.finditer(text):
        u = _units(_kata(run.group()))
        for k in range(len(u), 0, -1):
            piece = "".join(u[:k])
            surface = run.group()[:len(piece)]  # kept in hiragana: as katakana it may equal the term itself
            for ph in {phonemes(piece), phonemes(piece, True)}:
                if morae(ph) >= MIN_MORA and not piece.startswith("ー"):
                    out.append((surface, ph, "hrun" if k == len(u) else "hpre"))
    # vanilla: pyopenjtalk-plus's post-processing re-reads 何 / 風 / 方 through a Sudachi dictionary it builds anew
    # on every call (about 20 ms, 90% of the matcher's time); those readings do not matter for names
    words = [(w["string"], w["pron"].replace("’", ""), w["pos"])
             for w in pyopenjtalk.run_frontend(text, use_vanilla=True)]
    for i in range(len(words)):
        if words[i][2] in _SKIP_POS:
            continue
        for j in range(i, min(len(words), i + MAX_WORDS)):
            if _KATA.search(words[j][0]) or words[j][2] == "記号":
                break
            if words[j][2] in _SKIP_POS:
                continue
            surface = "".join(w[0] for w in words[i:j + 1])
            kind = "word" if _KANJI.search(surface) else "hira"
            for ph in {phonemes("".join(w[1] for w in words[i:j + 1]), True), *_sudachi(surface)}:
                if morae(ph) >= MIN_MORA:
                    out.append((surface, ph, kind))
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and _KATA_RUN.fullmatch(words[j][0]):
            j += 1
        if j == i:
            i += 1
            continue
        run = "".join(w[0] for w in words[i:j])
        for k in range(j, min(len(words), j + 2)):
            if words[k][2] in _SKIP_POS or not _KANJI.search(words[k][0]):
                break
            tail = words[j:k + 1]
            for ph in {phonemes("".join(w[1] for w in tail), True), *_sudachi("".join(w[0] for w in tail))}:
                out.append((run + "".join(w[0] for w in tail), phonemes(run) + ph, "mix"))
        i = j
    return out


@lru_cache(maxsize=1 << 16)
def _sudachi(surface: str) -> tuple[tuple[str, ...], ...]:
    if not _KANJI.search(surface):
        return ()
    from sudachipy import tokenizer

    morphemes = _sudachi_tok().tokenize(surface, tokenizer.Tokenizer.SplitMode.C)
    return (phonemes("".join(m.reading_form() for m in morphemes), True),)


@lru_cache(maxsize=1)
def _sudachi_tok():
    from sudachipy import dictionary

    return dictionary.Dictionary(dict="core").create()


# --- matcher ------------------------------------------------------------------------

LIMITS = {"kata": 0.08, "mix": 0.08, "part": 0.04, "word": 0.05, "hira": 0.0, "hrun": 0.0, "hpre": 0.0}
KANA_ON_WORD = 0.05   # a katakana term on a kanji word span (老苑 for ローエン)
FREQUENT = 0.20       # a frequent term on kata / mix spans; half that on part spans
NOISE_LINE = 4e-4     # background per-row rate at which a (term, surface) pair is dropped
MAX_COST = max(FREQUENT, *LIMITS.values(), KANA_ON_WORD)
SCREEN = 60           # rapidfuzz ratio (0-100) between phoneme strings a pair needs to be aligned at all


@dataclass(frozen=True)
class Term:
    ja: str
    kana: str      # the reading, or the katakana spelling of a kana name
    reading: bool  # whether ``kana`` is a reading (long vowels written as vowels) rather than a spelling


@dataclass(frozen=True)
class SoundHit:
    ja: str
    surface: str   # what the row wrote
    cost: float
    noise: float   # the pair's background per-row rate (0 when not in the noise table)


def available() -> bool:
    try:
        import pyopenjtalk  # noqa: F401
        import rapidfuzz  # noqa: F401
        import sudachipy  # noqa: F401
    except ImportError:
        return False
    return True


class SoundMatcher:
    def __init__(self, terms: Iterable[Term], noise: Mapping[tuple[str, str], float] | None = None):
        """``noise``: (term ja, span surface) -> background per-row rate; pairs at ``NOISE_LINE`` or above are
        dropped. Terms shorter than ``MIN_MORA`` are left out."""

        terms = list(terms)
        self.terms = [(t.ja, ph) for t in terms if morae(ph := phonemes(t.kana, t.reading)) >= MIN_MORA]
        self.names = {t.ja for t in terms}  # a span that is itself a term is that term, not another one misheard
        self.noise = dict(noise or {})
        self.by_len: dict[int, list[int]] = defaultdict(list)
        for k, (_, ph) in enumerate(self.terms):
            self.by_len[len(ph)].append(k)
        self.keys = [_key(ph) for _, ph in self.terms]

    @staticmethod
    def _passes(ja: str, c: float, kind: str, frequent: bool) -> bool:
        katakana = bool(_KATA.search(ja))
        if kind in ("hrun", "hpre") and not katakana:
            return False
        limit = KANA_ON_WORD if kind == "word" and katakana else LIMITS[kind]
        if frequent and kind in ("kata", "mix", "part"):
            limit = max(limit, FREQUENT / 2 if kind == "part" else FREQUENT)
        return c <= limit

    def match(self, texts: Sequence[str], frequent: frozenset[str] = frozenset()) -> dict[str, list[SoundHit]]:
        """text -> the terms its spans sound like (the cheapest span per term), for terms not written in it."""

        import numpy as np
        from rapidfuzz import fuzz, process

        row_spans = {t: spans(t) for t in set(texts)}
        by_len_span: dict[int, list[tuple[str, ...]]] = defaultdict(list)
        for ph in sorted({ph for sp in row_spans.values() for _, ph, _ in sp}):
            by_len_span[len(ph)].append(ph)
        scored: dict[tuple[str, ...], list[tuple[int, float]]] = {}
        for length, group in by_len_span.items():
            cand = [k for tl, ks in self.by_len.items() if KEEP * tl - 1 <= length <= tl * 1.4 + 1 for k in ks]
            if not cand:
                continue
            ckeys = [self.keys[k] for k in cand]
            for start in range(0, len(group), 4000):  # bounds the score matrix
                chunk = group[start:start + 4000]
                sim = process.cdist([_key(ph) for ph in chunk], ckeys, scorer=fuzz.ratio, dtype=np.uint8,
                                    score_cutoff=SCREEN, workers=-1)
                for a, b in zip(*np.nonzero(sim)):
                    k = cand[b]
                    if (c := cost(self.terms[k][1], chunk[a])) <= MAX_COST:
                        scored.setdefault(chunk[a], []).append((k, c))
        out: dict[str, list[SoundHit]] = {}
        for text, sp in row_spans.items():
            best: dict[str, SoundHit] = {}
            for surface, ph, kind in sp:
                if surface in self.names:
                    continue
                for k, c in scored.get(ph, ()):
                    ja = self.terms[k][0]
                    if ja in text or not self._passes(ja, c, kind, ja in frequent):
                        continue
                    if (noise := self.noise.get((ja, surface), 0.0)) >= NOISE_LINE:
                        continue
                    if ja not in best or c < best[ja].cost:
                        best[ja] = SoundHit(ja, surface, c, noise)
            out[text] = sorted(best.values(), key=lambda h: (h.cost, h.ja))
        return out


_CODES: dict[str, int] = {}


def _key(ph: tuple[str, ...]) -> str:  # one character per phoneme for rapidfuzz
    return "".join(chr(0x100 + _CODES.setdefault(p, len(_CODES))) for p in ph)
