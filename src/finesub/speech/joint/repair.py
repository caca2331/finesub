"""What the repairs of a cloud transcript share: finding the spots, and checking what comes back for them.

Units are dicts with `start`, `end`, `text`, `speaker` (absolute seconds), as `stitch` hands them on. Three kinds
of spot (`docs/joint-asr.md`「退化的修补」):

* a miss (35t, MAI2): a gap of at least GAP_MIN s between the units, each widened by PAD, in which Silero hears
  at least GATE s of speech;
* a repetition run (35t): one sequence of 1-4 units said REPEAT_MIN times in a row; FAST when one repetition
  takes under FAST_UNIT s, quicker than anyone talks, which `patch_fast` trims locally;
* a window with words but no speaker on any (35t), which the adapter flags.

What comes back for a spot -- another model's text (35t Live) or the backend's own re-ask -- only adds units
in that spot or replaces the run; `anchored` takes what Live wrote between the words either side of the spot,
`trim_echo` drops its edges where they repeat the neighbours, and `implausible` says why a text cannot be used.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Sequence

from ...subtitles.metrics import weighted_char_count
from .stitch import by_label, link

#: `--asr-repair`: `on` repairs at no quota (35t through Live), `quota` also re-asks 35t on its daily requests.
MODES = ("on", "quota", "off")
#: A speakerless 35t window labelled by voice: no reader has judged it, so it stays off (owner 2026-11-02: what no
#: acceptance has judged is off by default; internal, not an option).
SPEAKERLESS_BY_VOICE = False

GAP_MIN = 2.0
PAD = 0.3
GATE = 1.5
#: Gaps closer than this are re-asked as one span where the answer has times (MAI2, 35t's `quota`).
MERGE_WITHIN = 60.0
REPEAT_MIN = 10
FAST_UNIT = 0.12
#: A fast run keeps this many repetitions per second of it (at least FAST_KEEP_MIN).
FAST_PER_SECOND = 4.0
FAST_KEEP_MIN = 2
#: The collapse line: more weighted characters per second of Silero-heard speech is not speech.
MAX_CHARS_PER_SECOND = 25.0
#: An edge shorter than this (normalised characters) is not taken for an echo of the neighbour.
ECHO_MIN = 2
#: How much of each neighbour an echo is looked for in.
ECHO_SPAN = 10.0
#: The blocks a speaker's voice is taken from: a speaker's units split at pauses of PAUSE s, kept from BLOCK s.
PAUSE = 0.5
BLOCK = 1.0
#: `anchored`: aligned runs shorter than this (normalised characters) are chance -- CJK characters, Latin letters;
#: a side with words needs ANCHOR of its NEAR x MIN_MATCH characters next to the spot matched to anchor it; new
#: text ECHO of which the sides also hold is a variant of them, not new.
MIN_MATCH_CJK = 2
MIN_MATCH_LATIN = 4
ANCHOR = 0.5
NEAR = 4
ECHO = 0.5


def mode(value: str) -> str:
    value = value.strip().lower()
    if value not in MODES:
        raise ValueError(f"--asr-repair must be one of {', '.join(MODES)} (got {value!r})")
    return value


def key(text: str) -> str:
    """Letters and digits, NFKC, lower case: what two spellings of one unit share (`woah` and `whoa` do not)."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).lower() if unicodedata.category(ch)[0] in "LN")


# --- misses -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gap:
    start: float
    end: float
    voiced: float

    def as_list(self) -> list[float]:
        return [round(self.start, 2), round(self.end, 2)]


def gaps(units: Sequence[dict], total: float, voiced: Callable[[float, float], float]) -> list[Gap]:
    """The gaps Silero hears speech in, by start."""
    spans = sorted((u["start"] - PAD, u["end"] + PAD) for u in units)
    out, at = [], 0.0
    for a, b in [*spans, (total, total)]:
        if a - at >= GAP_MIN:
            heard = voiced(at, a)
            if heard >= GATE:
                out.append(Gap(at, a, heard))
        at = max(at, b)
    return out


def merged(found: Sequence[Gap], within: float = MERGE_WITHIN,
           longest: float = math.inf) -> list[tuple[float, float, list[Gap]]]:
    """Gaps less than `within` apart as one span, no longer than `longest` s: (start, end, its gaps)."""
    out: list[tuple[float, float, list[Gap]]] = []
    for g in found:
        if out and g.start - out[-1][1] < within and g.end - out[-1][0] <= longest:
            out[-1] = (out[-1][0], g.end, [*out[-1][2], g])
        else:
            out.append((g.start, g.end, [g]))
    return out


def neighbours(units: Sequence[dict], a: float, b: float) -> tuple[str, str]:
    """The normalised text of the units ending in the ECHO_SPAN s before `a`, and starting in the ECHO_SPAN s after
    `b`, each in time order."""
    left = sorted((u for u in units if a - ECHO_SPAN <= u["end"] <= a + PAD), key=lambda u: u["start"])
    right = sorted((u for u in units if b - PAD <= u["start"] <= b + ECHO_SPAN), key=lambda u: u["start"])
    return "".join(key(u["text"]) for u in left), "".join(key(u["text"]) for u in right)


def insert_at(units: Sequence[dict], start: float) -> int:
    """Where units filling a gap from `start` go in output order: after the unit that ends latest by then."""
    before = [(u["end"], i) for i, u in enumerate(units) if u["end"] <= start + PAD]
    return max(before)[1] + 1 if before else 0


def trim_echo(texts: Sequence[str], left: str, right: str) -> tuple[int, int]:
    """(first, end) of `texts` (one string per unit) left once the longest head that ends `left` and the longest
    tail that begins `right` are dropped, each at least ECHO_MIN characters (normalised)."""
    keys = [key(t) for t in texts]
    first = 0
    for k in range(len(keys), 0, -1):
        head = "".join(keys[:k])
        if len(head) >= ECHO_MIN and left.endswith(head):
            first = k
            break
    end = len(keys)
    for k in range(first, len(keys)):
        tail = "".join(keys[k:])
        if len(tail) >= ECHO_MIN and right.startswith(tail):
            end = k
            break
    return first, end


def _mapped(text: str) -> tuple[str, list[int]]:
    """`text` normalised character by character, with each character's index in `text`."""
    out, idx = [], []
    for i, ch in enumerate(text):
        for k in key(ch):
            out.append(k)
            idx.append(i)
    return "".join(out), idx


def _runs(x: str, y: str, least: int) -> list:
    return [m for m in difflib.SequenceMatcher(None, x, y, autojunk=False).get_matching_blocks() if m.size >= least]


def min_match(text: str) -> int:
    """MIN_MATCH_CJK when most of normalised `text` is wide (CJK, kana, hangul), else MIN_MATCH_LATIN: two
    letters of any alphabet match by chance."""
    wide = sum(unicodedata.east_asian_width(ch) in "WF" for ch in text)
    return MIN_MATCH_CJK if wide * 2 > len(text) else MIN_MATCH_LATIN


def echoed(text: str, around: str) -> bool:
    """Whether ECHO or more of normalised `text` is in runs of `around` (normalised)."""
    least = min_match(around or text)
    return bool(text) and sum(m.size for m in _runs(text, around, least)) >= ECHO * len(text)


def anchorable(side: str, least: int) -> bool:
    """Whether a side has enough characters for `anchored` to look for it."""
    return len(side) >= 2 * least


def anchored(answer: str, left: str, right: str) -> tuple[str, str | None]:
    """(new text, why not) from what Live wrote over a spot and the words either side of it (`left`, `right`:
    normalised, in time order): the right side is aligned first, then the left in what comes before it; the text
    between the last match on the left and the first on the right is new. Not when it is shorter than a match
    (`nothing_between`: Live wrote what 35t has), a side with words is not found next to the spot (`unheard`: no
    telling where the spot is in the answer), or it is an echo of the sides (`echo`)."""
    norm, idx = _mapped(answer)
    if not norm:
        return "", "empty"
    least = min_match(left + right or norm)
    in_right = _runs(norm, right, least)
    b = min((m.a for m in in_right), default=len(norm))
    in_left = _runs(norm[:b], left, least)
    a = max((m.a + m.size for m in in_left), default=0)
    near = NEAR * least
    for found, span in ((in_left, range(max(0, len(left) - near), len(left))), (in_right, range(min(near, len(right))))):
        hit = {k for m in found for k in range(m.b, m.b + m.size)} & set(span)
        if len(span) >= 2 * least and len(hit) < ANCHOR * len(span):  # the span is 2 x least iff the side is
            return "", "unheard"
    if b - a < least:
        return "", "nothing_between"
    if echoed(norm[a:b], left + right):
        return "", "echo"
    start, end = idx[a], idx[b - 1] + 1
    while end < len(answer) and not key(answer[end]) and not answer[end].isspace():
        end += 1  # the punctuation that closes it
    return answer[start:end].strip(), None


def implausible(text: str, voiced: float, language: str | None, *, script: bool = True) -> str | None:
    """Why a text that came back for a spot is not used, or None; the script is checked only with `script`."""
    from .mosspro import wrong_script

    if not key(text):
        return "empty"
    if weighted_char_count(text) > MAX_CHARS_PER_SECOND * max(voiced, 0.1):
        return "too_dense"
    if script and wrong_script([{"text": text}], language):
        return "script"
    return None


# --- repetition runs ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    first: int  # index of its first unit
    size: int  # units in one repetition
    times: int
    start: float
    end: float
    unit: str  # one repetition, normalised

    @property
    def stop(self) -> int:
        return self.first + self.size * self.times

    @property
    def fast(self) -> bool:
        return (self.end - self.start) / self.times < FAST_UNIT


def runs(units: Sequence[dict]) -> list[Run]:
    """Each run of one 1-4 unit sequence said REPEAT_MIN times or more in a row, in the order given (the survey's
    count: the longest cover wins at each position)."""
    keys = [key(u["text"]) for u in units]
    out, i = [], 0
    while i < len(keys):
        best = None
        for n in (1, 2, 3, 4):
            piece = keys[i:i + n]
            if len(piece) < n or not "".join(piece):
                continue
            times = 1
            while keys[i + times * n:i + (times + 1) * n] == piece:
                times += 1
            if times >= REPEAT_MIN and (best is None or times * n > best[0] * best[1]):
                best = (n, times)
        if best is None:
            i += 1
            continue
        n, times = best
        stop = i + n * times
        out.append(Run(i, n, times, units[i]["start"], max(u["end"] for u in units[i:stop]), "".join(keys[i:i + n])))
        i = stop
    return out


def patch_fast(units: Sequence[dict], run: Run) -> list[dict] | None:
    """A fast run's units cut to FAST_PER_SECOND repetitions per second (at least FAST_KEEP_MIN), its time shared
    evenly; None when that keeps them all."""
    keep = max(FAST_KEEP_MIN, math.ceil((run.end - run.start) * FAST_PER_SECOND))
    if keep >= run.times:
        return None
    one = units[run.first:run.first + run.size]
    step = (run.end - run.start) / (keep * run.size)
    out = []
    for k in range(keep * run.size):
        u = one[k % run.size]
        a = run.start + k * step
        out.append(dict(u, start=round(a, 3), end=round(a + step, 3)))
    return out


def longest_run(text: str, unit: str) -> int:
    """The most repetitions of `unit` (normalised) in a row anywhere in `text`."""
    if not unit:
        return 0
    found = re.findall(f"(?:{re.escape(unit)})+", key(text))
    return max((len(m) // len(unit) for m in found), default=0)


# --- voices -------------------------------------------------------------------------------------------------


def blocks(units: Sequence[dict]) -> list[dict]:
    """Each speaker's units (by start) split at pauses of PAUSE s; the blocks of BLOCK s or more, by start. Words are
    too short to take a voice from; a block is the piece `relink` embeds."""
    by: dict[str, list[dict]] = {}
    for u in sorted(units, key=lambda u: u["start"]):
        if u.get("speaker") is not None:
            by.setdefault(u["speaker"], []).append(u)
    out = []
    for speaker, us in by.items():
        cur = None
        for u in us:
            if cur is not None and u["start"] - cur["end"] < PAUSE:
                cur["end"] = max(cur["end"], u["end"])
                continue
            if cur is not None:
                out.append(cur)
            cur = {"start": u["start"], "end": u["end"], "speaker": speaker}
        if cur is not None:
            out.append(cur)
    return sorted((b for b in out if b["end"] - b["start"] >= BLOCK), key=lambda b: b["start"])


def relabel(known: Sequence[dict], new: Sequence[dict], lo: float, hi: float, voices) -> dict[str, str]:
    """A re-ask of [lo, hi]'s labels (in `new`) onto the transcript's (`known`): by the seconds both talk in the
    span (`stitch.link`), then by voice (`relink.Voices.match`); a label in neither is not in the map."""
    mapping = {c: p for p, c in link(by_label(known, lo, hi), by_label(new, lo, hi))}
    rest = {w["speaker"] for w in new if w["speaker"] is not None and w["speaker"] not in mapping}
    if rest:
        mapping.update(voices.match([b for b in blocks(new) if b["speaker"] in rest]))
    return mapping
