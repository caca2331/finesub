"""The rescue's pure decisions, on segment dicts (`{"start", "end", "speaker", "text"}`, seconds): no model, no GPU.

Why each rule exists and where its thresholds came from: `docs/joint-asr.md`「本地 MOSS 的救援」.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Sequence

import numpy as np

from ....subtitles.metrics import weighted_char_count
from ....text import punct_class
from ...postprocessing.segmentation import DEFAULT_SPLIT_PARAMS
from ..mosspro import LATIN, NON_LATIN
from ..relink import TWIN
from ..stitch import norm

#: A hole in the transcript this long (s), with GATE_VOICED s of Silero speech in it, is talk the pass skipped.
GAP = 30.0
GATE_VOICED = 3.0
#: Holes from SHORT_GAP s up are decoded as their own windows, with at least LEAD s of the transcript either side.
SHORT_GAP = 3.0
LEAD = 10.0
#: A window with no transcript before it starts this long before Silero's first speech in it.
ONSET_LEAD = 0.5
#: A window replaces the existing segments with at least REPLACE_COVER of their time in it, unless they lie wholly
#: more than REACH s outside the stretch it decoded.
REPLACE_COVER = 0.5
REACH = 0.5
#: A window's line says an existing segment again when at least TWIN_TIME of the line's time lies in it and at
#: least TWIN_TEXT of the line's letters and digits appear in its text, in order (owner 2026-11-20). A one- or
#: two-character reply inside a long segment passes the text share almost always: the time share is what holds it.
#: Chinese is not folded to simplified: a line written in the other script only fails to match.
TWIN_TIME = 0.8
TWIN_TEXT = 0.8
#: A clock check on gap windows, off (owner 2026-11-25: written in but kept off, to turn on if a window whose whole
#: clock runs off turns up): each existing segment inside the window is paired with the window's segment nearest it
#: saying the same (ANCHOR_SIMILAR of its letters and digits, in order), at any distance, by the nearer of their
#: start and end differences (a sentence the window split differently keeps one edge; a clock off moves both); the
#: window would be dropped when the lower median of those differences is over ANCHOR_TOL s. The offset is recorded
#: on every window's event either way.
CLOCK_CHECK = False
ANCHOR_TOL = 1.0
ANCHOR_SIMILAR = 0.7
#: A segment running on more than TAIL_MIN s past its last speech ends there, but no earlier than its text takes
#: at 1 / TAIL_SLACK of the source's median speaking rate.
TAIL_MIN = 1.0
TAIL_SLACK = 3.0
#: A segment repeating its last LOOP_SPAN tokens with a period of at most LOOP_PERIOD is looping.
LOOP_SPAN = 48
LOOP_PERIOD = 12
#: Wrong language: how much text a test needs, and the shares below which it is not the source language.
LANG_WORDS = 30
LANG_LETTERS = 40
EN_SHARE = 0.15
KANA_SHARE = 0.2
EN_COMMON = frozenset(
    "the a an and or but of to in on at for with from by as is are was were be been it its this that these those i "
    "you he she we they me him her us them my your his our their not no do does did have has had will would can "
    "could what who how why when where there here so if just like yeah okay oh all about up out".split())


def wrong_language(text: str, language: str | None) -> bool | None:
    """Whether `text` is not in `language`; None while it is too short to tell. English is told from other
    Latin-script languages by its common words, Japanese from Chinese by kana, the rest only by script (as
    `mosspro.wrong_script` checks MOSS Pro). A language it cannot test, or none, passes. Simplified against
    traditional Chinese is not a language error."""
    lang = (language or "").lower()
    if lang not in LATIN | NON_LATIN:
        return False
    if lang == "en":
        words = re.findall(r"[a-z']+", text.lower())
        if len(words) < LANG_WORDS:
            return None
        return sum(w in EN_COMMON for w in words) / len(words) < EN_SHARE
    # units: a run of Latin letters is one (a word), any other letter one each -- so a few English terms in Chinese
    # or Japanese talk do not outweigh the characters around them, as they would counted letter by letter
    names = [unicodedata.name(ch, "") for ch in re.sub(r"[A-Za-zÀ-ɏ]+", "A", text)
             if unicodedata.category(ch)[0] == "L"]
    if len(names) < LANG_LETTERS:
        return None
    latin = sum("LATIN" in n for n in names) / len(names)
    kana = sum("HIRAGANA" in n or "KATAKANA" in n for n in names) / len(names)
    if lang == "ja":
        return kana < KANA_SHARE
    if lang == "zh":
        return kana >= KANA_SHARE or latin > 0.5
    return latin < 0.5 if lang in LATIN else latin > 0.5


def periodic(body: Sequence[int]) -> bool:
    """Whether the last LOOP_SPAN tokens repeat with a period of at most LOOP_PERIOD."""
    if len(body) < LOOP_SPAN:
        return False
    tail = body[-LOOP_SPAN:]
    return any(all(tail[i] == tail[i - p] for i in range(p, LOOP_SPAN)) for p in range(1, LOOP_PERIOD + 1))


def gaps(segs: Sequence[dict], dur: float, least: float = GAP) -> list[tuple[float, float]]:
    """Stretches of at least `least` s that no segment covers, up to `dur`."""
    out, reach = [], 0.0
    for s in sorted(segs, key=lambda s: s["start"]):
        if s["start"] - reach >= least:
            out.append((reach, s["start"]))
        reach = max(reach, s["end"])
    if dur - reach >= least:
        out.append((reach, dur))
    return out


def window(segs: Sequence[dict], a: float, b: float, dur: float, cont: bool = False,
           onset: float | None = None) -> tuple[float, float]:
    """The audio [lo, hi] to decode for the gap [a, b] of the transcript `segs` (`dur` s long). Each side takes
    LEAD s of it and goes on out to the next segment boundary, so the window never cuts into a line: it starts at
    the last segment starting LEAD s or more before a (else the earliest segment before a) and ends at the first
    ending LEAD s or more after b (else the last one), pushed out past any segment straddling or just touching that
    point (someone talking over the neighbour, one stretch of talk the pass cut in two). A clip edge with no segment
    past it is the window's edge, except that a window with nothing before it starts ONSET_LEAD s before Silero's
    first speech in it (`onset`): what comes before is music or silence, which can throw the window's whole clock
    off. A continuation (`cont`, of a window that hit its token budget) starts at a, where that window stopped."""
    before = [s for s in segs if s["end"] <= a + 0.01]
    after = [s for s in segs if s["start"] >= b - 0.01]
    far = [s["start"] for s in before if s["start"] <= a - LEAD]
    lo = max(far) if far else min((s["start"] for s in before), default=a)
    far = [s["end"] for s in after if s["end"] >= b + LEAD]
    hi = min(far) if far else max((s["end"] for s in after), default=b)
    while (x := min((s["start"] for s in segs if s["start"] < lo - 0.01 < s["end"]), default=None)) is not None:
        lo = x
    while (y := max((s["end"] for s in segs if s["start"] < hi + 0.01 < s["end"]), default=None)) is not None:
        hi = y
    lo, hi = max(0.0, min(lo, a)), min(dur, max(hi, b))
    if cont:
        lo = a
    elif not before and onset is not None:
        lo = min(max(lo, onset - ONSET_LEAD), b)
    return lo, hi


def replace(segs: Sequence[dict], decoded: Sequence[dict], lo: float, hi: float) -> tuple[list[dict], list[dict]]:
    """`segs` with what the gap window [lo, hi] decoded in place of the context it decoded again, and the existing
    segments that went. All the window decoded goes in, so a line it wrote across the gap's edge, the gap's first or
    last words run on into the context, is neither lost nor kept twice. An existing segment goes when at least
    REPLACE_COVER of it lies in the window (an earlier window's segment may cross this one's edge) and it reaches
    into the stretch the window decoded, from its first start to its last end with REACH s either way: a window that
    hit its token budget or stopped early leaves the context it never got to. Context the window skipped mid-way,
    inside that stretch, goes without a new version."""
    if not decoded:
        return list(segs), []
    first = min(s["start"] for s in decoded) - REACH
    last = max(s["end"] for s in decoded) + REACH
    gone = [s for s in segs if s["start"] <= hi and s["end"] >= lo and s["start"] < last and s["end"] > first
            and min(s["end"], hi) - max(s["start"], lo) >= REPLACE_COVER * (s["end"] - s["start"])]
    out = {id(s) for s in gone}
    return sorted([*(s for s in segs if id(s) not in out), *decoded], key=lambda s: s["start"]), gone


def _letters(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if unicodedata.category(ch)[0] in "LN").lower()


def said_again(line: dict, old: dict) -> bool:
    """Whether the window's `line` is the existing segment `old` decoded again (TWIN_TIME, TWIN_TEXT)."""
    span = line["end"] - line["start"]
    inside = min(line["end"], old["end"]) - max(line["start"], old["start"])
    if span <= 0 or inside < TWIN_TIME * span:
        return False
    a, b = _letters(line["text"]), _letters(old["text"])
    found = sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return bool(a) and found >= TWIN_TEXT * len(a)


def twin(line: dict, replaced: Sequence[dict]) -> str | None:
    """The label of the replaced segments the window's `line` says again, when they have one label between them.
    A window's label with too little voice to tell who it is gives its lines these labels instead of none
    (`relink.unknown_rescues`): the line is the same words as before, so it is the same person. A replaced
    segment that an earlier window wrote and that itself says a segment before it again gives that one's label:
    the earlier window's label may be one too short to tell."""
    labels = {old.get(TWIN) or old["speaker"] for old in replaced
              if (old.get(TWIN) or old["speaker"]) is not None and said_again(line, old)}
    return labels.pop() if len(labels) == 1 else None


def clock_off(decoded: Sequence[dict], segs: Sequence[dict], lo: float, hi: float) -> tuple[int, float]:
    """How many existing segments inside the window [lo, hi] the window `decoded` again (anchors), and the lower
    median of their offsets (0.0 with none): what `CLOCK_CHECK` reads."""
    late = []
    for y in segs:
        t = norm(y["text"])
        if y["end"] <= lo or y["start"] >= hi or len(t) < 2:
            continue
        same = [x for x in decoded if len(norm(x["text"])) >= 2
                and difflib.SequenceMatcher(None, norm(x["text"]), t).ratio() >= ANCHOR_SIMILAR]
        if same:
            late.append(min(min(abs(x["start"] - y["start"]), abs(x["end"] - y["end"])) for x in same))
    late.sort()
    return len(late), (late[(len(late) - 1) // 2] if late else 0.0)  # the lower median: one chance match rejects nothing


def trim_tails(segs: Sequence[dict], spans: Sequence[tuple[float, float]]) -> tuple[list[dict], list[list[float]]]:
    """`segs` with late ends pulled in to their last Silero speech (`spans`), and [start, old end, new end] of
    each. Silero misses lines under game music, whose length is normal for their text: hence the rate floor."""
    def last_speech(s):
        return max((min(e, s["end"]) for a, e in spans if e > s["start"] and a < s["end"]), default=None)

    rows = [(s, last_speech(s), weighted_char_count(s["text"].strip())) for s in segs]
    rates = [w / (s["end"] - s["start"]) for s, last, w in rows
             if last is not None and s["end"] - last < 0.5 and s["end"] - s["start"] > 0.5 and w > 0]
    if not rates:
        return list(segs), []
    rate = float(np.median(rates))
    out, trimmed = [], []
    for s, last, w in rows:
        end = (s["end"] if last is None or s["end"] - last <= TAIL_MIN
               else min(s["end"], max(last, s["start"] + TAIL_SLACK * w / rate)))
        if end < s["end"] - 0.05:
            trimmed.append([round(s["start"], 2), round(s["end"], 2), round(end, 2)])
            s = {**s, "end": round(end, 2)}
        out.append(s)
    return out, trimmed


#: A Japanese segment with a run of more than this many weighted characters between marks (the splitter's longest
#: acceptable line) has lost its commas: open MOSS writes fewer of them as a pass goes on. Fixed, not scaled by
#: `--split-length-scale`: it only decides which segments are re-decoded.
UNMARKED_RUN = DEFAULT_SPLIT_PARAMS.chars_ok_hi
#: A re-decode whose text (marks aside) is less like the segment's than this lends it no marks.
GRAFT_SIMILAR = 0.9


def _mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def _bare(text: str) -> tuple[str, list[str]]:
    """(`text` without its punctuation marks, the marks right after each character left)."""
    chars, after = [], []
    for ch in text:
        if _mark(ch):
            if after:
                after[-1] += ch
        else:
            chars.append(ch)
            after.append("")
    return "".join(chars), after


def _latin(before: str, after: str) -> bool:
    """Whether a comma between these two characters sits in a number or Latin text (「1,000」「OK, so」), where a 、
    would not belong."""
    return before.isascii() and before.isalnum() and after.isascii()


def unmarked(text: str) -> bool:
    """Whether `text` has a run of more than UNMARKED_RUN weighted characters with no punctuation mark in it."""
    run = ""
    for ch in text + "。":
        if _mark(ch):
            if weighted_char_count(run) > UNMARKED_RUN:
                return True
            run = ""
        else:
            run += ch
    return False


def graft(text: str, heard: str) -> tuple[str, int, float]:
    """(Japanese `text` with a 、 wherever `heard` -- a re-decode of the same audio -- has a clause mark after a
    character both share, matched in order with the marks aside; the commas added; the similarity of the two). Only
    clause marks are lent (a window's own sentence end can fall mid-sentence: 「まあそれか。アルレッキーノ…」), and
    each is written 、 (the window, under the default Chinese prompt, may write ，); nothing goes after `text`'s last
    character, where `text` already has a mark, or inside a number or Latin text; no character of `text` changes, and
    below GRAFT_SIMILAR nothing is added."""
    mine, own = _bare(text)
    theirs, marks = _bare(heard)
    sm = difflib.SequenceMatcher(None, mine, theirs, autojunk=False)
    similar = sm.ratio()
    if similar < GRAFT_SIMILAR:
        return text, 0, similar
    add = {i + k for i, j, n in sm.get_matching_blocks() for k in range(n)
           if any(punct_class(ch) == "clause" for ch in marks[j + k]) and not own[i + k] and i + k < len(mine) - 1
           and not _latin(mine[i + k], mine[i + k + 1])}
    out, idx = [], -1
    for ch in text:
        out.append(ch)
        if not _mark(ch):
            idx += 1
            if idx in add:
                out.append("、")
    return "".join(out), len(add), similar
