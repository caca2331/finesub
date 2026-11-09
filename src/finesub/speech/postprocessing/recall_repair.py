"""The recall repair's decisions (docs/asr-stabilize.md, "补救：坍缩行与空档"):
which squeezed rows get their time back and which unclaimed speech becomes a
line. Pure: the second model's transcriptions come in as text
(`verification/recall_decode` makes them); no audio, no model.

Two repairs, in this order, on stabilize's rows (after the drop ladder and the
timeline contract):

1. Squeezed rows (`squeeze_repair.is_squeezed`: far too many characters for
   their span). Around each, windows the referee transcribes (`squeeze_windows`):
   the front [max(earlier rows' ends, start - FRONT), start] (when at least
   FRONT_MIN long), the gate [end, the plain stretch's new end + GATE_PAD]
   (rows the plain stretch reaches), the overlap gate [end, `_overlap_end` +
   GATE_PAD] (rows only the next row blocks), and the wide window. `verdict`
   reads them in order, the first that holds wins:
   - the front holds LATE of the row's first half: the start is late, not
     stretched (`late_start`);
   - the gate holds GATE of the row: stretched as the plain stretch would
     (`stretch`);
   - the overlap gate holds GATE, the wide window HEARD, and the rows the
     stretch would lie over under HEARD (not one speech written twice):
     stretched over the next row (`overlap`);
   - front and wide both under LOST: a suspected hallucination, left alone
     (`suspect`);
   - otherwise left alone (`between`, `heard`, `unstretched`).
   An overlap stretch is shown as the moved rows are (`shown`): over a row
   ending inside it, on the overlap lane; cut at the midpoint with a row
   running past its end.
2. Gaps: VAD speech of at least MIN_SEC no row covers, after step 1. The
   referee transcribes every gap and every row within NEAR s of one; Whisper
   transcribes the gaps whose referee text has content in the episode's
   language (`needs_whisper`). A gap is heard when Whisper's text has content
   too and the two readings share a longest common subsequence of at least
   AGREE of the shorter (`heard`). A heard gap holding a near row's missing
   speech moves that row (`placed`, `filled`); any other heard gap becomes a
   line: the referee's text over the gap, no words, `gap_fill: true` and
   Whisper's text beside it.

The constants are the accepted arms' (`sq-split`, `gap2r-a5` in the
maintainers' evaluation); changing one is a quality change to measure again.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from ...subtitles import line_groups
from ...subtitles.metrics import weighted_char_count
from ..languages import qwen_language_code
from ..recognition.segments import normalize_segment_timeline, uncovered_speech_pieces
from . import squeeze_repair as sr
from .recall_sounds import content, content_sounds, cover, lcs, missing, norm, substantive, unmatched
from .segmentation import split_params_of
from .squeeze_repair import SOURCE_FIELD, is_squeezed, stretch_plan, stretch_squeezed
from .stabilization import is_closing_shape, payload_audio_end

# Squeezed rows: the windows and the verdict's lines.
FRONT = 3.0
FRONT_MIN = 0.3
LATE = 0.5
GATE = 0.3
GATE_PAD = 0.3
LOST = 0.3
HEARD = 0.5
WIDE_MAX = 20.0
# Gaps.
MIN_SEC = 1.0
NEAR = 3.0
AGREE = 0.5
_EPS = 1e-6

Rows = list[dict]


def _overlap_end(i: int, rows: Rows, spans: list, intervals: list, keys: list, need: float, cap: float) -> float | None:
    """The new end of a flagged row the plain stretch leaves alone only because
    the next row starts at its end: grown through the VAD speech after it as
    the plain stretch grows a row, other rows aside, to `cap` at most. None for
    any other reason the plain stretch has (a closing shape, an echo, a row
    starting inside it, an earlier row over it, no speech of its own)."""
    row, (a, b) = rows[i], spans[i]
    if SOURCE_FIELD in row or is_closing_shape(str(row.get("text") or "")) or sr.echoes(i, keys):
        return None
    others = [s for j, s in enumerate(spans) if s is not None and j != i]
    following = [s[0] for s in others if s[0] >= a]
    limit = min((s for s in following if s >= b - _EPS), default=math.inf)
    if (limit > b + _EPS or any(s < b - _EPS for s in following)
            or any(s[0] < a and s[1] > b + _EPS for s in others)):
        return None
    own = sr.speech_within(intervals, a, b)
    if not own:
        return None
    have, reach, new_end = sum(y - x for x, y in own), max(y for _x, y in own), b
    for x, y in sorted(sr.speech_within(intervals, b, cap)):
        if have >= need - _EPS or x - reach > sr.MAX_SILENCE_BRIDGE_SEC + _EPS:
            break
        take = min(y - x, need - have)
        have, new_end, reach = have + take, x + take, y
    return new_end if new_end > b + _EPS else None


def squeeze_targets(payload: Mapping[str, object]) -> dict[int, dict]:
    """Per flagged row of `payload["segments"]`: the front window (or None), the
    plain stretch's new end (`plan`, or None), the overlap end (`overlap`, or
    None) and the wide window."""
    rows = payload["segments"]
    vad = payload["vad_timeline"]["intervals"]
    plan = stretch_plan(rows, vad, leave=is_closing_shape)
    intervals = sr.speech_intervals(vad)
    spans = [sr.row_span(r) for r in rows]
    keys = sr.echo_keys(rows, spans)
    tail = payload_audio_end(payload)
    out, before = {}, 0.0
    for i, row in enumerate(rows):
        if is_squeezed(row) and spans[i] is not None:
            a, b = spans[i]
            front = max(before, a - FRONT)
            need = weighted_char_count(str(row.get("text") or "")) / sr.TARGET_CHARS_PER_SEC
            wide = a + min(max(2 * need, need + 2.0), WIDE_MAX)
            over = None if i in plan else _overlap_end(i, rows, spans, intervals, keys, need, wide)
            reach = plan.get(i) or over
            wide = max(wide, reach + GATE_PAD if reach else 0.0)
            out[i] = {"front": (front, a) if a - front >= FRONT_MIN else None, "plan": plan.get(i), "overlap": over,
                      "wide": (a, min(wide, tail) if tail else wide)}
        if spans[i] is not None:
            before = max(before, spans[i][1])
    return out


def squeeze_windows(payload: Mapping[str, object]) -> list[dict]:
    """The windows to transcribe around the flagged rows: {"index", "kind"
    (front | gate | overlap | wide), "start", "end"}."""
    out = []
    r3 = lambda x: round(x, 3)
    for i, tg in squeeze_targets(payload).items():
        b = float(payload["segments"][i]["end"])
        if tg["front"]:
            out.append({"index": i, "kind": "front", "start": r3(tg["front"][0]), "end": r3(tg["front"][1])})
        for kind in ("plan", "overlap"):
            if tg[kind]:
                out.append({"index": i, "kind": "gate" if kind == "plan" else "overlap", "start": r3(b),
                            "end": r3(tg[kind] + GATE_PAD)})
        out.append({"index": i, "kind": "wide", "start": r3(tg["wide"][0]), "end": r3(tg["wide"][1])})
    return out


def verdict(row: dict, got: Mapping[str, str], planned: bool, lang: str, covered: str = "") -> dict:
    """What the referee says about one flagged row, from its windows' texts
    (`got`: kind -> text); `covered`: the text of the rows an overlap stretch
    would lie over."""
    text = str(row.get("text") or "")
    c = {k: cover(text, v, lang) for k, v in got.items()}
    if "front" in got:
        c["front_head"] = cover(text, got["front"], lang, head=True)
    if "overlap" in got:
        c["covered"] = cover(text, covered, lang)
    if c.get("front_head", 0.0) >= LATE:
        branch = "late_start"
    elif planned and c.get("gate", 0.0) >= GATE:
        branch = "stretch"
    elif ("overlap" in got and c["overlap"] >= GATE and c.get("wide", 0.0) >= HEARD
          and c["covered"] < HEARD):
        branch = "overlap"
    elif c.get("front", 0.0) < LOST and c.get("wide", 0.0) < LOST:
        branch = "suspect"
    elif planned:
        branch = "between"
    else:
        branch = "heard" if c.get("wide", 0.0) >= HEARD else "unstretched"
    return {"branch": branch, "cover": {k: round(v, 3) for k, v in c.items()}}


def repair_squeezed(payload: Mapping[str, object], heard: Sequence[dict], lang: str) -> tuple[Rows, list[dict]]:
    """`payload`'s rows with the flagged rows the referee lets through
    stretched, overlaps kept (`shown` displays them; the collapse ladder reads
    the whole stretch as its `S` proposal), and each flagged row's verdict.
    `heard`: `squeeze_windows`' windows, each with the referee's "text"."""
    rows, tgs = payload["segments"], squeeze_targets(payload)
    by: dict[int, dict[str, str]] = {}
    for w in heard:
        by.setdefault(w["index"], {})[w["kind"]] = w["text"]
    verdicts = []
    for i, tg in tgs.items():
        row = rows[i]
        covered = " ".join(str(r.get("text") or "") for j, r in enumerate(rows)
                           if j != i and tg["overlap"] and row["end"] - _EPS <= r["start"] < tg["overlap"])
        got = {k: t for k, t in by.get(i, {}).items()  # a window only where the row has its target
               if (k != "overlap" or tg["overlap"]) and (k != "gate" or tg["plan"])}
        v = verdict(row, got, tg["plan"] is not None, lang, covered)
        verdicts.append({"index": i, "start": row["start"], "end": row["end"], **v})
    go = {v["index"] for v in verdicts if v["branch"] == "stretch"}
    vad = payload["vad_timeline"]["intervals"]
    params = split_params_of(payload)
    segments, _ = stretch_squeezed(rows, vad, leave=is_closing_shape, params=params, only=go.__contains__)
    wide = {id(rows[v["index"]]): tgs[v["index"]]["overlap"] for v in verdicts if v["branch"] == "overlap"}
    intervals = sr.speech_intervals(vad)
    out = []
    for seg in segments:
        if id(seg) not in wide:
            out.append(seg)
            continue
        start, end = float(seg["start"]), float(seg["end"])
        row = dict(seg, end=wide[id(seg)], words=sr.spread_words(seg, start, wide[id(seg)], intervals),
                   **{SOURCE_FIELD: {"start": start, "end": end}})
        out.extend(sr.resplit(row, seg, vad, params))
    segments = sorted(out, key=lambda s: (float(s["start"]), float(s["end"])))
    segments, _ = normalize_segment_timeline(segments, audio_end=payload_audio_end(payload))
    return segments, verdicts


def overlap_sources(verdicts: Sequence[dict]) -> set[tuple[float, float]]:
    """The (start, end) of the rows `repair_squeezed` stretched over the next one: `shown`'s sources."""
    return {(v["start"], v["end"]) for v in verdicts if v["branch"] == "overlap"}


#: `shown`: a row lain over that ends at least INSIDE s before the moved piece
#: lies inside it (from the owner's listening, OD8 2026-11-03).
INSIDE = 0.05


def shown(rows: Rows, sources: set[tuple[float, float]], vad: Sequence[Mapping[str, float]],
          start_of: Callable[[dict, tuple[float, float]], float | None] | None = None) -> Rows:
    """`rows` with each piece of a moved row (its SOURCE_FIELD's (start, end)
    one of `sources`) and the first other row starting inside it shown as the
    owner's listening ruled (docs/asr-stabilize.md, "坍缩行"): a row ending at
    least INSIDE s before the moved row's last piece ends stays under it, the
    piece on the overlap lane (cut there, the piece would lose its tail; the
    whole row counts, not the splitter's piece); a row running past it starts
    at `start_of(row, source)` (in the ladder: where Whisper's pad re-decode of
    the moved row starts it) when that falls inside the row, else the two meet
    at the midpoint of the time they share, and the piece ends there unless it
    already ends earlier, unless the cut would leave either row squeezed
    (`squeeze_repair.is_squeezed`), when the two stay overlapped (owner
    2026-11-03; the midpoint uses that piece's end: a row starting in
    an earlier piece and running past the whole row is cut there, the later
    pieces staying over it on the lane; no such case in the evaluation). Every
    other overlap left goes on the lane (`laned`), a moved row lying inside a
    row that starts before it (a ladder span inside the next row, `crossed`)
    among them, whether or not that row runs past it: cut, that row would lose
    its head or its tail, as a row inside the moved one would (the evaluation's
    8 such rows all end inside the moved one). With nothing cut the rows keep
    their order."""
    intervals = sr.speech_intervals(vad)
    out = [dict(r) for r in rows]
    cut_any = False
    reach: dict[tuple[float, float], float] = {}
    for r in out:
        src = r.get(SOURCE_FIELD)
        if isinstance(src, dict):
            key = (src["start"], src["end"])
            reach[key] = max(reach.get(key, 0.0), float(r["end"]))
    for k in range(len(out)):
        src = out[k].get(SOURCE_FIELD)
        s = (src["start"], src["end"]) if isinstance(src, dict) else None
        if s not in sources:
            continue
        start, end = float(out[k]["start"]), float(out[k]["end"])
        j = next((j for j, b in enumerate(out) if j != k and b.get(SOURCE_FIELD) != src
                  and start <= float(b["start"]) < end), None)
        if j is None or float(out[j]["end"]) < reach[s] - INSIDE:
            continue
        b = out[j]
        b0, b1 = float(b["start"]), float(b["end"])
        cut = (b0 + min(end, b1)) / 2
        got = start_of(b, s) if start_of else None
        if got is not None and b0 < got < b1:
            cut = got
        if sr.is_squeezed(dict(out[k], end=min(cut, end))) or sr.is_squeezed(dict(b, start=cut)):
            continue  # cut, a row would flash by again: the defect the repair is for; overlapped instead
        if cut < end:
            out[k] = dict(out[k], end=cut, words=sr.spread_words(out[k], start, cut, intervals))
        out[j] = dict(b, start=cut, words=sr.spread_words(b, cut, b1, intervals))
        cut_any = True
    if not cut_any:
        return laned(list(rows), sources)
    return laned(sorted(out, key=lambda r: (float(r["start"]), float(r["end"]))), sources)


def laned(rows: Rows, sources: set[tuple[float, float]]) -> Rows:
    """`rows` with each piece of a moved row (its SOURCE_FIELD's (start, end)
    one of `sources`) that overlaps a row starting inside the moved row (not
    a piece of it) on its overlap lane (`line_groups.LANE_FIELD`, one per moved
    row, so two moved rows over each other both stay on screen): it stays
    over that row and the stages after are told the two are shown at once
    (docs/asr-stabilize.md, "坍缩行"). Every such piece, not only the one the
    row starts in: the splitter may cut the moved row inside it. Rows starting
    before the moved row's original start are not counted (a defect overlap
    it had before); the next row is, even when a ladder span (`crossed`)
    starts the moved row inside it."""
    def source(r: dict) -> tuple[float, float] | None:
        src = r.get(SOURCE_FIELD)
        return (src["start"], src["end"]) if isinstance(src, dict) else None

    out = list(rows)
    for k, r in enumerate(rows):
        s = source(r)
        if s not in sources:
            continue
        if any(source(b) != s and float(b["start"]) >= s[0]
               and float(b["start"]) < float(r["end"]) and float(r["start"]) < float(b["end"])
               for b in rows):
            out[k] = {**r, line_groups.LANE_FIELD: line_groups.lane_id(s)}
    return out


def gaps(payload: Mapping[str, object]) -> list[tuple[float, float]]:
    """The VAD speech no row of `payload` covers, at least MIN_SEC long, to the millisecond."""
    pieces = uncovered_speech_pieces(payload["vad_timeline"]["intervals"], payload["segments"])
    out = [(round(a, 3), round(b, 3)) for a, b in pieces]
    return [(a, b) for a, b in out if b - a >= MIN_SEC]


def near_rows(spans: Sequence[tuple[float, float]], rows: Rows) -> list[int]:
    """Indices of the rows within NEAR s of any of `spans`."""
    return [i for i, r in enumerate(rows) if any(r["start"] < b + NEAR and r["end"] > a - NEAR for a, b in spans)]


#: How the ASR step writes "not told, detected" (`transcribe.py` records
#: `language or "None"`, on the run and on each row).
_UNSET = ("", "None", "none", "auto")


def row_language(row: Mapping[str, object], episode: str | None) -> str | None:
    """A row's `lang`, or ``episode`` for a row without one (or with one meaning unset, as an undetected
    group's `"None"`)."""
    lang = str(row.get("lang") or "")
    return episode if lang in _UNSET else lang


def episode_language(payload: Mapping[str, object], rows: Rows) -> str | None:
    """The episode's language: what the ASR step was told (`asr_align.language`),
    else the rows' `lang` weighted by their content characters, a tie going to
    the language seen first; None without either."""
    told = str(((payload.get("metadata") or {}).get("asr_align") or {}).get("language") or "")
    if told not in _UNSET:
        return told
    weight: dict[str, int] = {}
    for row in rows:
        lang = str(row.get("lang") or "")
        if lang not in _UNSET:
            weight[lang] = weight.get(lang, 0) + len(content(norm(str(row.get("text") or ""), lang), lang))
    return max(weight, key=weight.get) if weight else None  # max keeps the first of equals


def needs_whisper(gap: Mapping[str, object], lang: str) -> bool:
    """The referee's half of `heard`: its text has content and is in the
    episode's language (the referee names the language, `qwen_language_code`).
    Only these gaps are worth Whisper's opinion."""
    return qwen_language_code(gap["language"]) == lang and substantive(norm(gap["qwen"] or "", lang), lang)


def heard(gap: Mapping[str, object], lang: str, agree: float = AGREE) -> bool:
    """Both models wrote content in the episode's language, and their readings agree."""
    if not needs_whisper(gap, lang):
        return False
    q, w = norm(gap["qwen"] or "", lang), norm(gap["whisper"] or "", lang)
    if not substantive(w, lang):
        return False
    sq, sw = content_sounds(q, lang), content_sounds(w, lang)
    return lcs(sq, sw) >= agree * min(len(sq), len(sw))


def _placed(gap: Mapping[str, object], rows: Rows, own: Mapping[int, dict], lang: str) -> int | None:
    """The near row whose missing speech the gap holds (index), nearest first;
    None: speech of its own. `own`: each near row's referee text over its own
    span."""
    sq = content_sounds(gap["qwen"], lang)
    near = [i for i, x in enumerate(rows) if x["start"] < gap["end"] + NEAR and x["end"] > gap["start"] - NEAR]
    gap_to = lambda i: max(rows[i]["start"] - gap["end"], gap["start"] - rows[i]["end"], 0.0)
    for i in sorted(near, key=gap_to):
        if qwen_language_code(own[i]["language"]) != lang and not is_squeezed(rows[i]):
            continue
        lacks = missing(content_sounds(rows[i].get("text"), lang), content_sounds(own[i]["qwen"], lang))
        if lacks and lcs(sq, lacks) >= 0.5 * len(sq):
            return i
    return None


def _retimed(row: dict, start: float, end: float, intervals: list[tuple[float, float]]) -> dict:
    """The row stretched over [start, end], its words spread over the VAD speech there; `gap_retime` is its old span."""
    old = row.get("gap_retime") or {"start": row["start"], "end": row["end"]}
    return {**row, "start": start, "end": end, "words": sr.spread_words(row, start, end, intervals), "gap_retime": old}


def fill_gaps(rows: Rows, heard_gaps: Sequence[dict], near: Sequence[dict], lang: str,
              intervals: list[tuple[float, float]], agree: float = AGREE) -> Rows:
    """`rows` with every heard gap filled or given to the near row that owns
    it, in time order. `heard_gaps`: {"start", "end", "qwen", "language",
    "whisper" (None when not asked)}; `near`: {"index", "qwen", "language"}
    per near row. Ownership is decided on `rows` as they are; a row owning
    several gaps moves once, over all of them, and no move crosses another
    row or a filled gap."""
    own = {r["index"]: r for r in near}
    add, claims = [], {}
    for g in heard_gaps:
        if not heard(g, lang, agree):
            continue
        i = _placed(g, rows, own, lang)
        if i is None:
            add.append({"start": g["start"], "end": g["end"], "text": g["qwen"], "words": [], "gap_fill": True,
                        "whisper": g["whisper"]})
        else:
            claims.setdefault(i, []).append(g)
    out = [dict(x) for x in rows]
    for i, claimed in claims.items():
        others = [x for k, x in enumerate(rows) if k != i] + add
        clear = lambda a, b: not any(x["start"] < b and x["end"] > a for x in others)
        a = min([rows[i]["start"]] + [g["start"] for g in claimed])
        b = max([rows[i]["end"]] + [g["end"] for g in claimed])
        ga, gb = min(g["start"] for g in claimed), max(g["end"] for g in claimed)
        if clear(a, b):
            out[i] = _retimed(out[i], a, b, intervals)
        elif clear(ga, gb):
            sr_ = content_sounds(rows[i].get("text"), lang)
            if len(sr_) - len(unmatched(sr_, content_sounds(own[i]["qwen"], lang))) < 0.25 * len(sr_):
                out[i] = _retimed(out[i], ga, gb, intervals)
    return sorted(out + add, key=lambda x: (x["start"], x["end"]))
