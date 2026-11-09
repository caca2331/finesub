"""The in-row fill's decisions (docs/asr-stabilize.md, "行内补漏"): which
rows the referee and Whisper hear again over their own span, and which
stretch of speech the referee hears there, the row lacks and Whisper hears
too goes into the row. Pure: the transcriptions come in as text
(`verification/recall_decode` makes them); no audio, no model.

Which rows (`candidates`): not squeezed (`squeeze_repair.is_squeezed`: their
speech is mostly elsewhere, not missing), not stretched by the collapse
repair (`SOURCE_FIELD`), and kept by the gate: a logistic score on the row's
own timing and Whisper's confidences (`gate_features`) at least GATE_THRESHOLD
(`--row-fill on`) or AUTO_THRESHOLD (`auto`: about half the decoding for about
three quarters of the gain).

Per row, on normalised text read as units (`recall_sounds.units`: kana for
Japanese, pinyin syllables for Mandarin, words where words are spaced):

* the row is left out when the referee's text is not in the episode's
  language, or holds under MATCHED of the row's own units (the two decoders
  disagree about the row: a mishear question, not a missing piece);
* the alignment is by contiguous blocks of units (`recall_sounds.unit_pairs`:
  at least BLOCK sounds, or WORD_BLOCK words); a candidate piece is a run of the referee's units
  the alignment leaves unmatched, cut to whole characters of its text, with at
  least SUBST content characters (`recall_sounds.content_words`: fillers off
  its ends, whole words where words are spaced), and with nothing of the row's left
  unmatched opposite it (LIGHT sounds aside): where the row says something
  else there, inserting would put two versions of one speech side by side;
* it goes in when Whisper's text over the span holds it (some run of Whisper's
  units the row leaves unmatched agrees with the piece by AGREE: longest
  common subsequence over the piece's length) and it is not a neighbour's
  speech bleeding in: a piece at the row's head is checked against as many
  units at the end of the row before, one at its tail against the start of
  the row after, one in the middle against the rows overlapping the row, and
  any piece against the whole row before when that row is still squeezed
  (`is_squeezed`: its speech is mostly past its span, often in this row's, so
  a phrase said twice would come out three times); an agreement by AGREE
  drops it.

Where it goes: between the characters its matched neighbours are. In a row
whose words join to its text (spaces aside) the piece is a word of its own,
`row_fill: true`, after the word its left neighbour lies in, timed over the gap
to the next word (often none: the word then has no duration); elsewhere it goes
into the text alone; where words are spaced, with one space either side. The
row keeps its span and lists its pieces under `row_fill` with Whisper's text.

The constants are the accepted arm's (`fuse-a5-fit70` in the maintainers'
evaluation); changing one is a quality change to measure again.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

from ..languages import qwen_language_code
from .recall_sounds import SUBST, UNSPACED, content_words, lcs, unit_pairs, units
from .squeeze_repair import SOURCE_FIELD, is_squeezed

AGREE = 0.5
MATCHED = 0.5
#: A long vowel, a geminate or a small kana: how two decoders spell one sound.
LIGHT = frozenset("ーっぁぃぅぇぉゃゅょゎ")
FIELD = "row_fill"

# The gate: a logistic regression fitted once on the maintainers' tier-2
# Japanese rows (13,349 rows, `od3_pregate --fitted 0.7`), its standard scaling
# folded into the weights, cut at the score that keeps 70% of the decoded audio
# there. EDGE is how far before and after a row its uncovered speech is summed.
EDGE = 1.0
GATE_FEATURES = ("dur", "maxword", "maxgap", "first_dur", "last_dur", "first_conf", "last_conf", "min_conf",
                 "mean_conf", "low_conf", "conf", "energy", "before", "after")
GATE_WEIGHTS = (0.06030643644341971, 4.242217628932519, 0.02212690658155613, 0.7234692954485873,
                -1.4050878686635764, -1.4167802965346765, -0.8224769917020947, -2.100293371553291,
                6.008602933015764, -0.31285941292063524, -1.3795264776720664, -0.025027999409931872,
                0.07478162867180223, 0.04673140188532466)
GATE_BIAS = -3.170196157390066
GATE_THRESHOLD = 0.24467164541409855
# `--row-fill auto`'s cut: on the maintainers' held-out sources (an offline replay) it keeps about 76% of the
# gate's net for about 53% of its decoding; the rows under it fill four times less per minute heard.
AUTO_THRESHOLD = 0.5

Rows = list[dict]
Span = tuple[float, float]


def _uncovered(spans: Sequence[Span], cover: Sequence[Span]) -> list[float]:
    """Lengths of the parts of `spans` that no interval of `cover` (sorted) covers."""
    out = []
    for x, y in spans:
        cur = x
        for s, e in cover:
            if e > cur and s < y:
                if s > cur:
                    out.append(s - cur)
                cur = max(cur, e)
        if y > cur:
            out.append(y - cur)
    return out


def gate_features(row: Mapping[str, object], vad: Sequence[Span], spans: Sequence[Span]) -> dict[str, float]:
    """The gate's features of `row`: its length (log), its longest word, the
    VAD speech in it no word covers, its first and last word (length,
    Whisper's confidence), the confidences' lowest, mean and share under 0.5,
    the row's confidence and VAD energy, and the VAD speech no row covers
    within EDGE s before its start and after its end. `vad`: the VAD
    intervals; `spans`: every row's span, sorted."""
    a, b = float(row["start"]), float(row["end"])
    clip = lambda x, y: [(max(x, s), min(y, e)) for s, e in vad if s < y and e > x]  # noqa: E731
    ws = [w for w in row.get("words") or [] if w.get("end") is not None]
    words = sorted((float(w["start"]), float(w["end"])) for w in ws)
    conf = [float(w.get("confidence", 1.0)) for w in ws] or [1.0]
    return {"dur": math.log(b - a + 0.1),
            "maxword": max((e - s for s, e in words), default=b - a),
            "maxgap": max(_uncovered(clip(a, b), words), default=0.0),
            "first_dur": words[0][1] - words[0][0] if words else b - a,
            "last_dur": words[-1][1] - words[-1][0] if words else b - a,
            "first_conf": conf[0], "last_conf": conf[-1], "min_conf": min(conf),
            "mean_conf": sum(conf) / len(conf), "low_conf": sum(c < 0.5 for c in conf) / len(conf),
            "conf": float(row.get("confidence") or 0.0),
            "energy": float(row.get("vad_weighted_energy_db") or 0.0),
            "before": sum(_uncovered(clip(a - EDGE, a), spans)), "after": sum(_uncovered(clip(b, b + EDGE), spans))}


def gate_score(features: Mapping[str, float]) -> float:
    z = GATE_BIAS + sum(w * features[f] for f, w in zip(GATE_FEATURES, GATE_WEIGHTS))
    return 1.0 / (1.0 + math.exp(-z))


def candidates(rows: Rows, intervals: Sequence[Mapping[str, object]], threshold: float = GATE_THRESHOLD) -> list[int]:
    """Indices of the rows worth hearing again: not squeezed, not stretched by
    the collapse repair, scored by the gate at least `threshold`. `intervals`:
    the VAD timeline's."""
    vad = [(float(v["start"]), float(v["end"])) for v in intervals]
    spans = sorted((float(r["start"]), float(r["end"])) for r in rows)
    return [k for k, r in enumerate(rows) if not is_squeezed(r) and SOURCE_FIELD not in r
            and gate_score(gate_features(r, vad, spans)) >= threshold]


def _runs(n: int, matched: set[int]) -> list[tuple[int, int]]:
    """The contiguous [a, b) runs of range(n) outside `matched`."""
    out, a = [], None
    for k in range(n + 1):
        if k < n and k not in matched:
            a = k if a is None else a
        elif a is not None:
            out.append((a, k))
            a = None
    return out


def _agrees(piece: list[str], other: list[str]) -> bool:
    return bool(piece) and lcs(piece, other) >= AGREE * len(piece)


def _whole(qs: list[tuple[str, int, int]], a: int, b: int) -> tuple[int, int] | None:
    """The run [a, b) of units cut to the characters all of whose units are in it; None when none are."""
    while a < b and a > 0 and qs[a][1] == qs[a - 1][1]:
        a += 1
    while b > a and b < len(qs) and qs[b - 1][1] == qs[b][1]:
        b -= 1
    return (a, b) if a < b else None


def _units(text: object, lang: str) -> list[str]:
    return [u for u, _, _ in units(str(text or ""), lang)]


def _word_chars(row: Mapping[str, object]) -> list[int | None] | None:
    """For every character of the row's text, the index of the word it lies in
    (None for spaces, which stabilize puts between merged segments and words do
    not carry); None when the words do not join to the text without its spaces."""
    words = row.get("words") or []
    texts = [str(w.get("word", w.get("text", ""))) for w in words]
    text = str(row.get("text") or "")
    if not words or "".join(texts).replace(" ", "") != text.replace(" ", ""):
        return None
    owner, k, used = [], 0, 0
    for c in text:
        if c == " ":
            owner.append(None)
            continue
        while k < len(texts) and used >= len(texts[k].replace(" ", "")):
            k, used = k + 1, 0
        owner.append(k)
        used += 1
    return owner


def pieces(row: Mapping[str, object], heard: Mapping[str, object], prev: Mapping[str, object] | None,
           nxt: Mapping[str, object] | None, overlapping: Sequence[Mapping[str, object]], lang: str,
           tally: Counter | None = None) -> list[dict]:
    """The row's pieces that pass every gate: the referee's text, its position
    (`after`: the row character it follows, -1 at the head) and Whisper's
    text. `heard`: {"qwen", "language", "whisper"} over the row's span, Whisper
    None when it was not asked (`needs_whisper` says when it must be). `tally`
    counts what each precondition and gate left out."""
    tally = tally if tally is not None else Counter()
    if qwen_language_code(heard["language"]) != lang:
        tally["rows_other_language"] += 1
        return []
    text, qwen = str(row.get("text") or ""), str(heard["qwen"] or "")
    rs, qs = units(text, lang), units(qwen, lang)
    r, q = [u for u, _, _ in rs], [u for u, _, _ in qs]
    match = unit_pairs(q, r, lang)
    if not r or len(match) < MATCHED * len(r):
        tally["rows_unmatched"] += 1
        return []
    whisper = extra = None  # read once a piece reaches the gate that needs it
    out = []
    for a, b in _runs(len(q), {i for i, _ in match}):
        left = max((j for i, j in match if i < a), default=None)
        right = min((j for i, j in match if i >= b), default=None)
        cut = _whole(qs, a, b)
        raw = qwen[qs[cut[0]][1]:qs[cut[1] - 1][2]] if cut else ""
        if len(content_words(raw, lang)) < SUBST:
            tally["pieces_short"] += 1
            continue
        opposite = r[0 if left is None else left + 1:len(r) if right is None else right]
        if any(s not in LIGHT for s in opposite):
            tally["pieces_substitution"] += 1
            continue
        piece = q[cut[0]:cut[1]]
        if extra is None:
            if heard["whisper"] is None:
                raise ValueError("a piece reached the gate that reads Whisper's text, which was not asked for")
            whisper = str(heard["whisper"])
            w = _units(whisper, lang)
            extra = [w[x:y] for x, y in _runs(len(w), {i for i, _ in unit_pairs(w, r, lang)})]
        if not any(_agrees(piece, e) for e in extra):
            tally["pieces_no_second_opinion"] += 1
            continue
        if left is None:
            near = _units(prev["text"], lang)[-len(piece):] if prev else []
        elif right is None:
            near = _units(nxt["text"], lang)[:len(piece)] if nxt else []
        else:
            near = None
        if (_agrees(piece, near) if near is not None else
                any(_agrees(piece, _units(o["text"], lang)) for o in overlapping)) or (
                prev is not None and is_squeezed(prev) and _agrees(piece, _units(prev["text"], lang))):
            tally["pieces_neighbour"] += 1
            continue
        out.append({"text": raw, "after": -1 if left is None else rs[left][2] - 1, "whisper": whisper})
    tally["pieces"] += len(out)
    return out


def needs_whisper(row: Mapping[str, object], heard: Mapping[str, object], lang: str) -> bool:
    """Whether a piece of the row passes every gate the referee's text decides
    alone, so that Whisper's text over the row is read (`pieces` with an empty
    Whisper text counts such a piece as `pieces_no_second_opinion`)."""
    tally = Counter()
    pieces(row, {**heard, "whisper": ""}, None, None, [], lang, tally)
    return bool(tally["pieces_no_second_opinion"])


def placed(row: dict, found: Sequence[dict], lang: str) -> dict:
    """The row with its pieces inserted, in their order (where words are
    spaced, with one space either side)."""
    if not found:
        return dict(row)
    text, words = str(row.get("text") or ""), [dict(x) for x in row.get("words") or []]
    owner = _word_chars(row)
    inserts = []  # (text position, word position or None, start, end, order, piece)
    for n, p in enumerate(found):
        if owner is None:
            pos = p["after"] + 1 if p["after"] >= 0 else 0
            while pos and lang not in UNSPACED and pos < len(text) and not text[pos].isspace():
                pos += 1  # after the whole word, its punctuation with it
            inserts.append((pos, None, None, None, n, p))
            continue
        wl = owner[p["after"]] if p["after"] >= 0 else None
        if wl is None:  # the head: before the first word
            pos, at = min(i for i, o in enumerate(owner) if o is not None), 0
        else:  # after the word the left neighbour lies in (a piece inside one word goes after it)
            pos, at = max(i for i, o in enumerate(owner) if o == wl) + 1, wl + 1
        a = words[at - 1]["end"] if at else row["start"]  # over the gap between that word and the next
        b = words[at]["start"] if at < len(words) else row["end"]
        inserts.append((pos, at, a, max(a, b), n, p))
    new = dict(row)
    for pos, at, a, b, _, p in sorted(inserts, key=lambda x: (x[0], x[4]), reverse=True):
        text = (text[:pos] + p["text"] + text[pos:] if lang in UNSPACED else
                f"{text[:pos].rstrip()} {p['text']} {text[pos:].lstrip()}".strip())
        if at is not None:
            words.insert(at, {"start": a, "end": b, "word": p["text"], FIELD: True})
    new["text"] = text
    if owner is not None:
        new["words"] = words
    new[FIELD] = [{"text": p["text"], "whisper": p["whisper"]} for p in found]
    return new


def fill_rows(rows: Rows, heard: Mapping[int, Mapping[str, object]], lang: str,
              tally: Counter | None = None) -> Rows:
    """`rows` with the pieces of the rows in `heard` (index -> {"qwen",
    "language", "whisper"}) inserted. The neighbour gates read `rows` as they
    are, before any insertion: the rows before and after in this list and the
    rows overlapping in time."""
    out = []
    for k, row in enumerate(rows):
        if k not in heard:
            out.append(dict(row))
            continue
        overlapping = [x for m, x in enumerate(rows) if m != k and x["start"] < row["end"] and x["end"] > row["start"]]
        found = pieces(row, heard[k], rows[k - 1] if k else None, rows[k + 1] if k + 1 < len(rows) else None,
                       overlapping, lang, tally)
        out.append(placed(row, found, lang))
    return out
