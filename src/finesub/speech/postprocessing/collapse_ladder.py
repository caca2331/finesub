"""The collapse repair's ladder (docs/asr-stabilize.md, "坍缩行"): for each
squeezed row, proposals in turn, the first that holds wins. Pure: Whisper's
decodes and the referee's texts come in as data (`verification/recall_decode`
makes them); no audio, no model.

Proposers, in the order a row takes them (`STEPS`):

* `pad`: Whisper's decode, the ASR stage's own path, of the VAD intervals the
  row's span grown by PAD s each side touches (`windows`);
* `S`: the referee-checked stretch (`recall_repair.repair_squeezed`): the
  row's new span joined over the pieces the splitter cut it into;
* `iso`: Whisper over the intervals the row's own span touches;
* `ctx`: Whisper over the intervals of the row's wide window grown by
  CTX_BEFORE s in front and CTX_AFTER s behind.

`iso` and `ctx` are read only for a row `pad` gave nothing (its decode missed
the row or left it where it was). A Whisper proposal is the span of the
decoded words that go to the row (the rows in the window aligned to the
decode by blocks of units; at least HEARD of the row's units), its start no
earlier than the earlier rows' ends and its end no later than the second next
row's start (`branch`); one that starts past the next row's start (`crossed`)
is taken only when no proposer gave a span that stays before it. Rows and
decodes are read as `recall_sounds.units`: sounds in Japanese and Mandarin,
words in English. `S` and the referee's on-time start (`late`) are the
stretch's own reads (`recall_repair`: sounds, letters in English).

Every proposal passes the same guards (`guarded`): not still squeezed, not an
echo, not a later start on a row the referee heard starting on time, not over
rows that already say HEARD of it; and the referee's presence check: its text
over the span holds HEARD of the row (`present`; a span too short for its
check windows is not heard). Whisper's proposals then have a first or last
word Whisper drew out past LONG s per sound cut back to TYPICAL s per sound
(`pinned`, per language: PIN) and pass the guards again.

The chosen span moves the row (`moved`): its words laid over the VAD speech in
the span, the row through the splitter alone, and a piece lying over the next
row shown as the owner's listening ruled (`recall_repair.shown`): over a row
ending inside it, on the overlap lane; cut with a row running past its end,
where Whisper's `pad` decode of the moved row starts that row (else at the
midpoint).

The constants are the accepted arm's (`lad-heard-pin2-xl` in the maintainers'
evaluation); changing one is a quality change to measure again.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..recognition.segments import GAP_KEEP_REAL_MAX_SEC, normalize_segment_timeline
from . import squeeze_repair as sr
from .recall_repair import HEARD, shown, squeeze_targets
from .recall_sounds import UNSPACED, sounds, unit_cover, unit_pairs, units
from .segmentation import split_params_of
from .stabilization import payload_audio_end

STEPS = ("pad", "S", "iso", "ctx")
KINDS = ("pad", "iso", "ctx")
#: The kinds read only when `pad` gave nothing (its branch one of these).
RETRIED_AFTER = ("whisper_miss", "same")
TAKEN = ("whisper", "crossed")
PAD = 2.0
CTX_BEFORE = 10.0
CTX_AFTER = 5.0
#: Both ends within this of the old ones: the decode left the row where it was.
SAME = 0.05
#: The presence check's windows: the span less TRIM at either end, and a
#: window under MIN_CLIP is not heard.
TRIM = 0.25
MIN_CLIP = 0.1
#: The pin, per language: a word over LONG s per sound (`recall_sounds.sounds`:
#: letters in English) is cut to TYPICAL s per sound, (LONG, TYPICAL) the 99th
#: percentile and the median of Whisper's words there (Japanese: the
#: evaluation's tier 2; English: the maintainers' zh / en English runs; Mandarin
#: takes Japanese's, as verified). Other languages take the measured pair of
#: their kind of sound (`pin_of`): Cantonese's hanzi and Korean's hangul are
#: syllables, as in Mandarin; the rest are letters, as in English.
PIN = {"ja": (0.58, 0.12), "zh": (0.58, 0.12), "en": (0.56, 0.08)}
_EPS = 1e-6

Rows = list[dict]
Span = tuple[float, float]


def _group(vad: Sequence[Span], segs: Sequence[Span], a: float, b: float) -> list[int] | None:
    """The VAD intervals (indices) touching [a, b], whole, grown until no aligned
    segment lies across the group's edges; None when they touch no speech."""
    hit = None
    while True:
        now = [k for k, (s, e) in enumerate(vad) if e > a and s < b]
        if not now or now == hit:
            return hit
        hit = now
        lo, hi = vad[hit[0]][0], vad[hit[-1]][1]
        a = min([a] + [s for s, e in segs if e > lo + _EPS and s < hi - _EPS])
        b = max([b] + [e for s, e in segs if e > lo + _EPS and s < hi - _EPS])


def windows(payload: Mapping[str, object], rows: Rows, kinds: Sequence[str] = KINDS,
            only: set[int] | None = None) -> list[dict]:
    """Whisper's decode groups of `kinds` around the squeezed rows of `rows`
    (those in `only`, when given): {"index", "kind", "start", "end",
    "intervals" (whole VAD intervals), "tail" (the real audio allowed after the
    last, as the ASR stage allows it)}. `payload` carries the aligned segments
    and the VAD timeline."""
    vad = sorted((float(v["start"]), float(v["end"])) for v in payload["vad_timeline"]["intervals"])
    segs = [(float(s["start"]), float(s["end"])) for s in payload["segments"]]
    out = []
    for kind in kinds:
        for i, tg in squeeze_targets({**payload, "segments": rows}).items():
            if only is not None and i not in only:
                continue
            a, b = float(rows[i]["start"]), float(rows[i]["end"])
            span = {"iso": (a, b), "pad": (a - PAD, b + PAD),
                    "ctx": (tg["wide"][0] - CTX_BEFORE, tg["wide"][1] + CTX_AFTER)}[kind]
            if hit := _group(vad, segs, *span):
                gap = vad[hit[-1] + 1][0] - vad[hit[-1]][1] if hit[-1] + 1 < len(vad) else None
                out.append({"index": i, "kind": kind, "start": vad[hit[0]][0], "end": vad[hit[-1]][1],
                            "intervals": [{"start": vad[k][0], "end": vad[k][1]} for k in hit],
                            "tail": GAP_KEEP_REAL_MAX_SEC if gap is None else max(0.0, min(gap, GAP_KEEP_REAL_MAX_SEC))})
    return out


def _words(segments: Sequence[Mapping[str, object]]) -> list[dict]:
    return [{"word": str(x.get("word") or ""), "start": float(x["start"]), "end": float(x["end"])}
            for s in segments for x in s.get("words") or []]


def assigned_words(rows: Rows, i: int, window: Mapping[str, object], lang: str) -> list[dict] | None:
    """The window's decoded words (`window["segments"]`) that go to row `i`, in
    order: every row in the window, in order, aligned to the decode by
    contiguous blocks of units, each decoded word going to one row; None
    unless they hold HEARD of the row's units."""
    words = _words(window["segments"])
    seq, own = [], []
    for j, r in enumerate(rows):
        if j == i or (float(r["end"]) > window["start"] and float(r["start"]) < window["end"]):
            us = [u for u, _, _ in units(str(r.get("text") or ""), lang)]
            seq += us
            own += [j] * len(us)
    n = own.count(i)
    joined, owner = "", []
    gap = "" if lang in UNSPACED else " "  # the stage's words carry no spaces
    for k, w in enumerate(words):
        owner += [k] * (len(w["word"]) + len(gap))
        joined += w["word"] + gap
    ws = units(joined, lang)
    mp = [(p, ws[q]) for p, q in unit_pairs(seq, [u for u, _, _ in ws], lang)] if seq and ws else []
    got = sorted({k for p, (_, a, b) in mp if own[p] == i for k in owner[a:b]})
    if not n or sum(own[p] == i for p, _ in mp) < HEARD * n:
        return None
    return [words[k] for k in got]


def branch(rows: Rows, i: int, window: Mapping[str, object] | None, lang: str) -> tuple[str, Span | None]:
    """What one Whisper decode (None: the row's window touches no speech) says
    about row `i`: (branch, the span of the words that go to it, to the
    millisecond, or None). `whisper` and `crossed` propose a span."""
    got = assigned_words(rows, i, window, lang) if window else None
    if got is None:
        return "whisper_miss", None
    g0, g1 = min(w["start"] for w in got), max(w["end"] for w in got)
    found = (round(g0, 3), round(g1, 3))
    row = rows[i]
    a, b = float(row["start"]), float(row["end"])
    spans = [sr.row_span(r) for r in rows]
    before = max((sp[1] for sp in spans[:i] if sp is not None), default=0.0)
    s = max(g0, before)
    e = min(g1, float(rows[i + 2]["start"])) if i + 2 < len(rows) else g1
    covered = " ".join(str(r.get("text") or "") for j, r in enumerate(rows) if j != i and b - _EPS <= float(r["start"]) < e)
    if abs(s - a) < SAME and abs(e - b) < SAME:
        return "same", found
    if i + 1 < len(rows) and s >= float(rows[i + 1]["start"]):
        return "crossed", found
    if e - s < b - a - SAME:
        return "shrunk", found
    if covered and unit_cover(str(row.get("text") or ""), covered, lang) >= HEARD:
        return "written_twice", found
    return "whisper", found


def retried(rows: Rows, decoded: Mapping[tuple[str, int], Mapping[str, object]], lang: str) -> set[int]:
    """The squeezed rows whose `pad` decode (in `decoded`) gave nothing: the
    only ones `pick` reads `iso` and `ctx` for, so the only ones to decode them for."""
    return {i for i, r in enumerate(rows) if sr.is_squeezed(r)
            and branch(rows, i, decoded.get(("pad", i)), lang)[0] in RETRIED_AFTER}


def _proposed(rows: Rows, i: int, got: Span) -> Span:
    """A taken Whisper span as the row would move to it."""
    before = max((float(r["end"]) for r in rows[:i]), default=0.0)
    return max(got[0], before), (min(got[1], float(rows[i + 2]["start"])) if i + 2 < len(rows) else got[1])


def guarded(rows: Rows, i: int, span: Span | None, echo: bool, late: bool, lang: str) -> tuple[Span | None, str | None]:
    """`span` for row `i` through the shared guards: (span, None), or (None, the guard that stopped it)."""
    if span is None:
        return None, None
    row, (s, e) = rows[i], span
    if e - s <= _EPS or sr.is_squeezed(dict(row, start=s, end=e)):
        return None, "still_squeezed"
    if echo:
        return None, "echo"
    if late and s > float(row["start"]) + _EPS:
        return None, "late_start"
    said = " ".join(str(r.get("text") or "") for j, r in enumerate(rows)
                    if j != i and float(r["end"]) > s + _EPS and float(r["start"]) < e - _EPS)
    if said and unit_cover(str(row.get("text") or ""), said, lang) >= HEARD:
        return None, "written_twice"
    return (s, e), None


def _joined(stretched: Rows) -> dict[Span, Span]:
    """Each row the stretch moved (by its old span, `SOURCE_FIELD`) and its new span, its pieces joined."""
    new: dict[Span, Span] = {}
    for r in stretched:
        src = r.get(sr.SOURCE_FIELD)
        if isinstance(src, dict):
            key, se = (src["start"], src["end"]), (float(r["start"]), float(r["end"]))
            s, e = new.get(key, se)
            new[key] = (min(s, se[0]), max(e, se[1]))
    return new


def proposals(rows: Rows, stretched: Rows, verdicts: Sequence[Mapping[str, object]],
              decoded: Mapping[tuple[str, int], Mapping[str, object]], lang: str) -> list[dict]:
    """Per squeezed row: {"index", "echo", "late", and per proposer its span and
    guard, per Whisper kind its branch}. `stretched` and `verdicts`: the
    referee-checked stretch's rows (overlaps kept) and verdicts; `decoded`:
    (kind, index) -> the window with its finished "segments"."""
    keys = sr.echo_keys(rows, [sr.row_span(r) for r in rows])
    joined = _joined(stretched)
    late = {v["index"] for v in verdicts if v["branch"] == "late_start"}
    out = []
    for i, row in enumerate(rows):
        if not sr.is_squeezed(row):
            continue
        rec = {"index": i, "echo": sr.echoes(i, keys), "late": i in late}
        rec["S"], rec["S_guard"] = guarded(rows, i, joined.get((row["start"], row["end"])), rec["echo"], rec["late"], lang)
        for k in KINDS:
            b, got = branch(rows, i, decoded.get((k, i)), lang)
            rec[f"{k}_branch"] = b
            rec[k], rec[f"{k}_guard"] = guarded(rows, i, _proposed(rows, i, got) if b in TAKEN else None,
                                                rec["echo"], rec["late"], lang)
        out.append(rec)
    return out


def _key(span: Span) -> Span:
    return round(span[0], 3), round(span[1], 3)


def to_hear(recs: Sequence[Mapping[str, object]]) -> list[tuple[int, Span]]:
    """The spans the presence check asks the referee about: each distinct
    proposal of each row, to the millisecond, the start at 0 at the earliest."""
    out, seen = [], set()
    for r in recs:
        for p in STEPS:
            if r[p] is not None:
                s, e = _key(r[p])
                if (r["index"], s, e) not in seen:
                    seen.add((r["index"], s, e))
                    out.append((r["index"], (round(max(s, 0.0), 3), e)))
    return out


def present(row: Mapping[str, object], span: Span, heard: str, lang: str) -> bool:
    """The presence check: the referee's text over the span holds HEARD of the
    row, and the span's check windows (it, and it less TRIM at either end, to
    the millisecond) are at least MIN_CLIP long."""
    s, e = span
    cut = TRIM * (e - s)
    if any(round(b, 3) - round(a, 3) < MIN_CLIP for a, b in ((s, e), (s + cut, e), (s, e - cut))):
        return False
    return unit_cover(str(row.get("text") or ""), heard, lang) >= HEARD


def pin_of(lang: str) -> tuple[float, float]:
    """(LONG, TYPICAL) for `lang`: its own, else the measured pair of its kind of sound (see PIN)."""
    return PIN.get(lang) or PIN["zh" if lang in ("yue", "ko") else "en"]


def pinned(span: Span, words: Sequence[Mapping[str, object]], lang: str, rule: str) -> Span:
    """`span` with its first ("head") or last ("tail") word cut to TYPICAL per
    sound when it runs past LONG per sound (`pin_of(lang)`). A span of one stretched
    word pinned at both ends comes out reversed; the guards' still-squeezed
    check drops it."""
    s, e = span
    long, typical = pin_of(lang)
    w = words[0] if rule == "head" else words[-1]
    n = max(len(sounds(w["word"], lang)), 1)
    if (w["end"] - w["start"]) / n <= long:
        return s, e
    return (max(s, w["end"] - typical * n), e) if rule == "head" else (s, min(e, w["start"] + typical * n))


def settle(rows: Rows, recs: list[dict], heard: Mapping[tuple[int, Span], str],
           decoded: Mapping[tuple[str, int], Mapping[str, object]], lang: str) -> list[dict]:
    """`recs` after the presence check (`heard`: (index, span) -> the referee's
    text over it, for every span `to_hear` listed) and the pins."""
    out = []
    for r in recs:
        r, i = dict(r), r["index"]
        for p in STEPS:
            if r[p] is not None:
                s, e = _key(r[p])
                span = (round(max(s, 0.0), 3), e)
                r[p] = span if present(rows[i], span, heard[(i, span)], lang) else None
                r[f"{p}_guard"] = "check" if r[p] is None else r[f"{p}_guard"]
        for rule in ("head", "tail"):
            for p in KINDS:
                if r[p] is not None:
                    words = assigned_words(rows, i, decoded[(p, i)], lang)
                    r[p], guard = guarded(rows, i, pinned(r[p], words, lang, rule), r["echo"], r["late"], lang)
                    r[f"{p}_guard"] = guard and f"{rule}:{guard}"
        out.append(r)
    return out


def pick(r: Mapping[str, object]) -> str | None:
    """The first proposer of STEPS with a span for the row; a `crossed` span
    only when none gave one that stays before the next row; `iso` and `ctx`
    only when `pad`'s branch is one of RETRIED_AFTER."""
    for take in (False, True):
        for p in STEPS:
            if p in KINDS[1:] and r["pad_branch"] not in RETRIED_AFTER:
                continue
            if p in KINDS and r[f"{p}_branch"] == "crossed" and not take:
                continue
            if r[p] is not None:
                return p
    return None


def decide(rows: Rows, recs: Sequence[Mapping[str, object]]) -> tuple[dict[int, Span], list[dict]]:
    """The span each row takes and, per squeezed row, its verdict: the proposer
    (or why it has none: the first proposer's guard) and every proposer's guard."""
    spans, verdicts = {}, []
    for r in recs:
        p = pick(r)
        if p:
            spans[r["index"]] = r[p]
        why = p or next((f"{q}:{r[f'{q}_guard']}" for q in STEPS if r.get(f"{q}_guard")), "no_span")
        verdicts.append({"index": r["index"], "start": rows[r["index"]]["start"], "end": rows[r["index"]]["end"],
                         "branch": why, **({"span": [round(x, 3) for x in r[p]]} if p else {}),
                         "guards": {q: r[f"{q}_guard"] for q in STEPS if r.get(f"{q}_guard")}})
    return spans, verdicts


def moved(payload: Mapping[str, object], rows: Rows, spans: Mapping[int, Span],
          decoded: Mapping[tuple[str, int], Mapping[str, object]] | None = None, lang: str = "") -> Rows:
    """`rows` with each row `i` in `spans` moved to its span (words laid over the
    VAD speech there, the row through the splitter alone), a piece lying over
    the next row shown by `recall_repair.shown`, with the start Whisper's
    `pad` decode in `decoded` gives the row it lies over."""
    vad = payload["vad_timeline"]["intervals"]
    intervals, params = sr.speech_intervals(vad), split_params_of(payload)
    out = []
    for i, row in enumerate(rows):
        if i not in spans:
            out.append(row)
            continue
        s, e = spans[i]
        new = dict(row, start=s, end=e, words=sr.spread_words(row, s, e, intervals),
                   **{sr.SOURCE_FIELD: {"start": row["start"], "end": row["end"]}})
        out.extend(sr.resplit(new, row, vad, params))
    out, _ = normalize_segment_timeline(sorted(out, key=lambda r: (float(r["start"]), float(r["end"]))),
                                        audio_end=payload_audio_end(payload))
    index = {(rows[i]["start"], rows[i]["end"]): i for i in spans}

    def pad_start(b: Mapping[str, object], source: Span) -> float | None:
        window = (decoded or {}).get(("pad", index[source]))
        jb = next((q for q, x in enumerate(rows) if x.get("text") == b.get("text")
                   and abs(float(x["start"]) - float(b["start"])) < 1e-6), None)
        got = assigned_words(rows, jb, window, lang) if window and jb is not None else None
        return min(w["start"] for w in got) if got else None

    return shown(out, set(index), vad, pad_start)

