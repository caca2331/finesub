"""Gemini 3.5 Transcribe ("35t") through `/v1beta/interactions`, free keys only.

The source is cut into windows of at most WINDOW s at Silero pauses and sent
one after another: a window ends at a pause in the REACH s before WINDOW, the
next starts at one 60-90 s before that end (`cuts`), the previous window's
words lowering the bar where they break. Each answer is read in
output order (35t emits by speaker: B's backchannel comes after A's sentence
although it is earlier) and repaired:

1. bad times: a word with end < start or longer than BAD_DUR, or one starting
   more than BACK_JUMP before the previous word of the same speaker ended,
   starts a run that is shifted back as a whole when a matching opposite jump
   closes it; otherwise only the bad word is fixed (`fix_times`);
2. degeneration: a 1-4 word sequence repeated REPEAT times in a row, or an
   answer whose status is not `completed`, splits the window in two halves,
   each strictly shorter, cut at pauses OVERLAP / 2 s either side of the
   middle (`_split`);
3. early stop: a window whose last word ends more than GAP s before the window
   does continues from a pause CONTINUE_BACK s before that word, at least
   MIN_ADVANCE past the window start (`_continued`); the last window continues
   once. A window read once is never the continuation (a cached half of a
   degenerate window is not read again as the tail).

Keys: the `GEMINI_FREE` pool, one key until Gemini answers that its requests
per day are spent, then the next (owner: never the paid pool automatically).
Rules of the API: `docs/gemini35-transcribe.md`.
"""

from __future__ import annotations

import json
import mimetypes
import re
import statistics
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from finesub import api_keys
from finesub.api_http import (
    CONNECT_TIMEOUT_SECONDS,
    RETRYABLE_TRANSPORT,
    api_http_client,
    is_retryable_http_error,
    retryable_status,
)
from finesub.media.ffmpeg import extract_upload_mp3, probe_media_duration
from finesub.reporting import bind_reporter, current_reporter

from ...subtitles.metrics import weighted_char_count
from . import GEMINI_35T, cuts
from . import live as live_client
from . import repair as R
from .cache import CacheMiss, CallCache, audio_fingerprint
from .model import Transcript, Unit, turns_from_units
from .mosspro import SpeechSpans
from .quota import PACIFIC, KeyLedger, QuotaExhausted, pick_key
from .relink import VOICEPRINTS, Voiceprints, Voices
from .stitch import stitch

if TYPE_CHECKING:
    from .stage import AlignerUse

MODEL = "gemini-3.5-transcribe"
UPLOAD = "https://generativelanguage.googleapis.com/upload/v1beta/files"
INTERACTIONS = "https://generativelanguage.googleapis.com/v1beta/interactions"

WINDOW = 540.0
#: The least overlap of two windows (`cuts.OVERLAP`), as the cached answers' identity records it.
OVERLAP = cuts.OVERLAP[0]
#: A window's end looks for a pause this far before WINDOW, as every cut's does (owner 2026-11-22); a cached window
#: is still found ending up to CACHED_REACH s before it, the reach windows were cut by from 2026-11-21 to 11-22.
REACH = cuts.REACH
CACHED_REACH = 60.0
MIN_ADVANCE = 120.0
#: An early stop goes on from a pause this far before the last word: at least, at most.
CONTINUE_BACK = (10.0, 20.0)
GAP = 30.0
BAD_DUR = 3.0
BACK_JUMP = 0.5
BIG_JUMP = 3.0
REPEAT = 20
#: 35t takes BCP-47 codes; the task's language maps here, the rest is auto-detected (as the experiment ran).
LANGUAGE_CODES = {"ja": "ja-JP"}


class Gemini35tError(RuntimeError):
    pass


# --- reading an answer ------------------------------------------------------------------------------------


def _sec(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value) if isinstance(value, (int, float)) else float(str(value).rstrip("s"))


def raw_words(raw: dict, offset: float) -> list[dict]:
    """Words in the order 35t emitted them, times absolute."""
    out = []
    for step in raw.get("steps", []) or []:
        for content in step.get("content", []) or []:
            for ann in content.get("annotations", []) or []:
                if ann.get("type") == "word_info":
                    out.append({"start": _sec(ann.get("start_offset")) + offset,
                                "end": _sec(ann.get("end_offset")) + offset,
                                "speaker": ann.get("speaker"), "text": ann.get("text", "")})
    return out


def answer_status(raw: dict) -> str:
    return str(raw.get("status", ""))


def _median_duration(words: list[dict]) -> float:
    ds = [w["end"] - w["start"] for w in words if 0 < w["end"] - w["start"] <= BAD_DUR]
    return statistics.median(ds) if ds else 0.3


def _entering(words: list[dict], i: int) -> tuple[float, bool] | None:
    """(jump, inside_word) when word i is a bad point, else None."""
    w = words[i]
    if w["end"] < w["start"] or w["end"] - w["start"] > BAD_DUR:
        return w["end"] - w["start"], True
    if i and words[i - 1]["speaker"] == w["speaker"] and w["start"] < words[i - 1]["end"] - BACK_JUMP:
        return w["start"] - words[i - 1]["end"], False
    return None


def fix_times(words: list[dict]) -> tuple[list[dict], list[dict]]:
    """Repaired copy, still in output order, and one event per bad point."""
    ws = [dict(w) for w in words]
    d = _median_duration(ws)
    events = []
    i = 0
    while i < len(ws):
        hit = _entering(ws, i)
        if hit is None:
            i += 1
            continue
        jump, inside = hit
        w = ws[i]
        ev = {"at": i, "text": w["text"], "start": w["start"], "end": w["end"], "jump": round(jump, 2)}
        if abs(jump) < BIG_JUMP:
            if w["end"] < w["start"]:
                w["end"] = w["start"] + 0.01
            ev["fix"] = "clamp"
            events.append(ev)
            i += 1
            continue
        # The run: words after the entering point, up to the first opposite jump of similar size in the same turn.
        first = i + 1 if inside else i
        leave = None  # (index of the word the run ends with, inside_word, right anchor)

        def opposite(x: float, n: int) -> bool:
            return x * jump < 0 and abs(abs(x) - abs(jump)) <= n * d + 1.0

        j = first
        while j < len(ws) and ws[j]["speaker"] == w["speaker"]:
            if j > first and opposite(ws[j]["start"] - ws[j - 1]["end"], j - first):
                leave = (j - 1, False, ws[j]["start"])  # the jump between j-1 and j: j is back in place
                break
            if opposite(ws[j]["end"] - ws[j]["start"], j - first + 1):
                leave = (j, True, ws[j]["end"])  # the jump inside j: its start is in the run, its end is fine
                break
            j += 1
        shift = -jump + d
        if leave is not None:
            last, leave_inside, right = leave
            shifted = []
            for k in range(first, last + 1):
                s = ws[k]["start"] + shift
                e = ws[k]["end"] if (leave_inside and k == last) else ws[k]["end"] + shift
                shifted.append((k, s, e))
            new_end = w["start"] + d if inside else None
            ok = all(e <= right + 1e-6 and s <= e for _, s, e in shifted)
            if inside:
                ok = ok and (not shifted or shifted[0][1] >= new_end - 1e-6)
            if ok:
                if inside:
                    w["end"] = new_end
                for k, s, e in shifted:
                    ws[k]["start"], ws[k]["end"] = round(s, 3), round(e, 3)
                ev.update(fix="shift", run=[ws[k]["text"] for k, _, _ in shifted], shift=round(shift, 2))
                events.append(ev)
                i = last + 1
                continue
        # Only the bad word: keep its start unless that start disagrees with the same speaker's previous word.
        if inside:
            prev = ws[i - 1] if i and ws[i - 1]["speaker"] == w["speaker"] else None
            if prev is not None and w["start"] < prev["end"] - BACK_JUMP:
                w["start"] = w["end"] - d
                ev["fix"] = "word-keep-end"
            else:
                w["end"] = w["start"] + d
                ev["fix"] = "word-keep-start"
        else:
            ev["fix"] = "unmatched"
        events.append(ev)
        i += 1
    return ws, events


def longest_repeat(words: list[dict]) -> tuple[int, str]:
    """Most consecutive repetitions of any 1-4 word sequence (in start order, as the experiment counted)."""
    texts = [w["text"].strip("。、，,.!?！？ ") for w in sorted(words, key=lambda w: w["start"])]
    best, what = 1, ""
    for n in range(1, 5):
        i = 0
        while i + n <= len(texts):
            unit = texts[i:i + n]
            k = 1
            while texts[i + k * n:i + (k + 1) * n] == unit:
                k += 1
            if k > best:
                best, what = k, "".join(unit)
            i += max(1, (k - 1) * n) if k > 1 else 1
    return best, what


# --- calling ----------------------------------------------------------------------------------------------


def _client():
    import httpx

    return api_http_client(httpx.Timeout(1800.0, connect=CONNECT_TIMEOUT_SECONDS))


def _upload(client, path: Path, key: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "audio/mpeg"
    size = path.stat().st_size
    start = client.post(UPLOAD, content=json.dumps({"file": {"display_name": path.name}}).encode(), headers={
        "x-goog-api-key": key, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json"})
    start.raise_for_status()
    session = start.headers["X-Goog-Upload-URL"]
    done = client.post(session, content=path.read_bytes(), headers={
        "Content-Length": str(size), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
    done.raise_for_status()
    return done.json()["file"]["uri"], mime


def request_body(uri: str, mime: str, *, language: str | None) -> dict:
    config: dict = {"mode": {"type": "verbatim", "timestamp_granularities": ["word"], "diarization_mode": "speaker"}}
    if language:
        config["language_codes"] = [language]
    return {"model": MODEL, "input": [{"type": "audio", "uri": uri, "mime_type": mime}],
            "generation_config": {"transcription_config": config}}


_QUOTA_ID = re.compile(r"quotaid[\"'\s:]+[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)


def quota_kind(text: str) -> str | None:
    """"daily", "minute", "unclear" (a quota error that does not say which) or None (not a quota error).

    By the structured `quotaId` when there is one -- a per-minute 429 also opens with "You exceeded your
    current quota" -- else by a per-day phrase (the LLM layer reads Gemini's 429s the same way,
    `llm.client.classify_quota_error`). The daily wall has also come back as HTTP 200 with only that
    sentence in an `error` (`docs/gemini35-transcribe.md`), which is "unclear"."""
    ids = [m.group(1).lower() for m in _QUOTA_ID.finditer(text)]
    if ids:
        return "daily" if any("perday" in q for q in ids) else "minute"
    lowered = text.lower()
    if "requests per day" in lowered or "per day" in lowered:
        return "daily"
    if "exceeded your current quota" in lowered or "resource_exhausted" in lowered:
        return "unclear"
    return None


#: Tries per window request (and per upload), and the wait after a transient failure that names no wait of its
#: own; a transient failure is the one `api_http` classifies so, or a 2xx that is not a usable answer.
ATTEMPTS = 4
TRANSIENT_WAIT = 60.0


class Caller:
    """One window's request on the first key with requests left today; a key Gemini reports spent is
    marked for (key, model) and the next one tried."""

    def __init__(self, *, ledger: KeyLedger | None = None, entries=None, sleep: Callable[[float], None] = time.sleep,
                 announce: Callable[[], None] = lambda: None):
        self.ledger = ledger or KeyLedger(GEMINI_35T, MODEL, PACIFIC)
        self._entries = entries
        self._sleep = sleep
        self._announce = announce  # before each request, once a key is in hand

    def entries(self):
        if self._entries is None:
            self._entries = api_keys.resolve_pool(api_keys.GEMINI_FREE_POOL, api_keys.read_dotenv())
        return self._entries

    def __call__(self, audio: Path, *, language: str | None, seconds: float) -> dict:
        while True:
            entry = pick_key(self.entries(), self.ledger, what="gemini-35t (GEMINI_FREE)")
            self._announce()
            try:
                body = self._once(audio, entry.key, language=language)
            except QuotaExhausted:
                self.ledger.mark_exhausted(entry)
                continue
            self.ledger.record(entry, seconds)
            return body

    def _once(self, audio: Path, key: str, *, language: str | None) -> dict:
        with _client() as client:
            for attempt in range(ATTEMPTS):
                try:
                    uri, mime = _upload(client, audio, key)
                    break
                except Exception as error:
                    if attempt == ATTEMPTS - 1 or not is_retryable_http_error(error):
                        raise
                    self._sleep(TRANSIENT_WAIT)
            body = json.dumps(request_body(uri, mime, language=language)).encode()
            unclear = False
            for attempt in range(ATTEMPTS):
                kind = None
                try:
                    r = client.post(INTERACTIONS, content=body, headers={"x-goog-api-key": key, "Content-Type": "application/json"})
                except RETRYABLE_TRANSPORT as error:  # no answer at all: a reset, a dropped proxy, a timeout
                    status, text = None, f"{type(error).__name__}: {error}"
                else:
                    status, answer = r.status_code, _answer(r)
                    if answer is not None and not answer.get("error"):
                        return answer
                    # An error is never an answer: returned, it would be cached as this window's transcript.
                    text = json.dumps(answer["error"], ensure_ascii=False) if answer is not None else r.text
                    kind = quota_kind(text)
                    if kind == "daily" or (kind == "unclear" and unclear):
                        # Unclear twice in a row on one key: the day is spent (the LLM layer's strike gate, smaller).
                        raise QuotaExhausted(text[:300])
                if attempt == ATTEMPTS - 1:
                    break
                wait = re.search(r"retry in (\d+(?:\.\d+)?)s", text)
                if kind is not None:  # per-minute, or unclear the first time
                    unclear = unclear or kind == "unclear"
                    self._sleep(float(wait.group(1)) + 5 if wait else TRANSIENT_WAIT)
                    continue
                # No answer, a retryable status ("high demand" and the like), or a 2xx that carries an error or
                # is not a JSON object at all (a gateway's page, a body cut off): transient.
                if status is None or status < 400 or retryable_status(status):
                    self._sleep(TRANSIENT_WAIT)
                    continue
                raise Gemini35tError(f"HTTP {status}: {text[:2000]}")
            raise Gemini35tError(f"no answer after retries: {text[:2000]}")


def _answer(r) -> dict | None:
    """A 2xx body as a JSON object; None for an error status or a body that is not one."""
    if r.status_code >= 400:
        return None
    try:
        answer = r.json()
    except ValueError:
        return None
    return answer if isinstance(answer, dict) else None


# --- repairs (`docs/joint-asr.md`「退化的修补」) ---------------------------------------------------------------

#: Live hears a spot with the words 35t has either side of it -- from the first ending in the CTX s before it to
#: the last starting in the CTX s after it (the spot's own edge where a side has none), plus EDGE s -- and only what
#: it writes between those words is new (`repair.anchored`); heard alone, Live wrote variants of the neighbours and
#: half sentences, which lost the reader vote. When that call fails, or Live wrote nothing over the window though
#: 35t has words on both sides of the spot, or (a run) Live's answer does not hold the sides, the spot alone is heard,
#: LIVE_PAD s either side out to a Silero pause (`_Spot.deaf`, `cuts.padded`). The wide window needs no pause: its
#: edges are 35t's own words', EDGE s outside them.
CTX = 5.0
EDGE = 0.3
LIVE_PAD = 1.0
#: The cache directories of Live's answers, beside the windows' (each its own identity file): the wide window
#: (named by its audio), and the spot alone.
LIVE_WIDE_CACHE = f"{GEMINI_35T}-live-ctx"
LIVE_CACHE = f"{GEMINI_35T}-live"
#: New text for a miss that weighs less (`weighted_char_count`) is not inserted: 1-3 character fragments.
MIN_NEW = 3.0
#: `--asr-repair quota`: re-asks on the daily requests, their own cache directory; a span is padded by
#: REASK_PAD s (and widened to REASK_MIN s); a speakerless window is asked again in two halves of at most REASK_HALF s
#: (`_split`); at most REASK_CAP calls per source, the rest are left to Live.
REASK_CACHE = f"{GEMINI_35T}-repair"
REASK_PAD = 10.0
REASK_MIN = 30.0
REASK_HALF = 300.0
REASK_CAP = 4


class _Spot:
    """One span asked of Live: a miss (`gap`) or a repetition run at talking speed (`run`), with the units around it
    (for a run, the transcript without the run)."""

    def __init__(self, kind: str, start: float, end: float, around: list[dict], total: float, *, gap=None, run=None):
        self.kind, self.start, self.end, self.gap, self.run = kind, start, end, gap, run
        left = sorted((u for u in around if start - CTX <= u["end"] <= start + R.PAD), key=lambda u: u["start"])
        right = sorted((u for u in around if end - R.PAD <= u["start"] <= end + CTX), key=lambda u: u["start"])
        self.left, self.right = "".join(R.key(u["text"]) for u in left), "".join(R.key(u["text"]) for u in right)
        self.lo = max(0.0, min((u["start"] for u in left), default=start) - EDGE)
        self.hi = min(total, max((u["end"] for u in right), default=end) + EDGE)
        self.text: str | None = None
        self.heard: str | None = None  # "wide", or "alone"
        self.error: str | None = None
        self.wide_error: str | None = None  # the wide call's failure
        self.wide_unused: str | None = None  # why its answer says Live did not hear the window (`deaf`)
        self.alone_error: str | None = None  # why the spot alone, asked after a wide answer, did not come back

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.start:.1f}-{self.end:.1f}"

    def deaf(self) -> str | None:
        """Why a wide answer says Live did not hear the window -- empty though 35t has words on both sides, or for
        a run (35t's words itself) empty or not holding the sides -- so the spot alone is asked; else None."""
        if self.heard != "wide":
            return None
        _, why = R.anchored(self.text, self.left, self.right)
        if why == "empty" and (self.run is not None or (self.left and self.right)):
            return "empty"
        return why if why == "unheard" and self.run is not None else None

    def new_text(self) -> tuple[str, str | None, bool]:
        """What Live wrote for the spot itself, why not when it wrote nothing new, and whether to check its script:
        not when the wide answer held a side long enough to look for (Live wrote the source's language there; an
        English quotation in a Chinese talk is the source's too)."""
        if self.heard != "wide":
            return self.text, None, True
        text, why = R.anchored(self.text, self.left, self.right)
        least = R.min_match(self.left + self.right)
        return text, why, not (R.anchorable(self.left, least) or R.anchorable(self.right, least))

    def unused(self, new: list[dict], least: float = 0.0) -> str | None:
        """Why the units left after `_trimmed` are not used: none, an echo of the sides (heard alone; the wide
        answer was checked in `anchored`), or less than `least` weighted characters."""
        said = "".join(u["text"] for u in new)
        if not new or (self.heard == "alone" and R.echoed(R.key(said), self.left + self.right)):
            return "echo"
        return "short" if weighted_char_count(said) < least else None

    def entry(self, **more) -> dict:
        out = {"repair": "miss" if self.kind == "gap" else "repeat", "span": [round(self.start, 2), round(self.end, 2)],
               "via": "live", "fed": "mix"}
        if self.heard is not None:
            out["heard"] = self.heard
            if self.heard == "wide" or self.wide_unused is not None:  # the wide answer's cache name
                out["window"] = [round(self.lo, 2), round(self.hi, 2)]
        if self.wide_error is not None:
            out["wide_failed"] = self.wide_error
        if self.wide_unused is not None:
            out["wide_unused"] = self.wide_unused
        if self.alone_error is not None:
            out["alone_failed"] = self.alone_error
        return {**out, **more}


def _timed(text: str, a: float, b: float, speaker, heard: list[tuple[float, float]], neighbours: list[dict],
           align) -> tuple[list[dict], str]:
    """Live's text as units over [a, b], and which timing: the forced aligner's (`align`, `--asr-aligner`; its
    audio stops at the `neighbours`' speech) where it gives them, else S5b on `heard`, Silero's speech in the span."""
    from . import moss_timing

    segment = {"start": a, "end": b, "text": text, "speaker": speaker}
    if align is not None:
        got = align(segment, neighbours)
        if got:
            return [dict(u, speaker=speaker) for u in got], "aligner"
    return moss_timing.warp(segment, heard), "S5b"


def _voiced_spans(pcm, a: float, b: float) -> list[tuple[float, float]]:
    from . import moss_timing

    lo, hi = int(a * live_client.SR), int(b * live_client.SR)
    return [(x + a, y + a) for x, y in moss_timing.speech_spans(moss_timing.frame_probabilities(pcm[lo:hi]))]


def _padded(a: float, b: float, total: float, pauses: cuts.Pauses, known: Sequence[tuple[float, float]],
            breaks: cuts.Breaks = ()) -> tuple[float, float]:
    """[a, b] padded by REASK_PAD s, widened to REASK_MIN s, each edge out to a pause (`cuts.padded`) no further
    than one window allows (the pads themselves are always given)."""
    short = max(0.0, REASK_MIN - (b - a + 2 * REASK_PAD)) / 2
    return cuts.padded(a - short, b + short, REASK_PAD, pauses, total, known, longest=WINDOW, breaks=breaks)


def _split(s: float, e: float, pauses: cuts.Pauses, known: Sequence[tuple[float, float]],
           longest: float | None = None, breaks: cuts.Breaks = ()) -> list[tuple[float, float]]:
    """[s, e] as two overlapping halves, each strictly shorter: the first ends `d` s after the middle and the second
    starts `d` s before it (`d` = OVERLAP / 2 -- the fixed halves before 2026-11-21 -- in a window long enough that
    each half is a second shorter, else a third of the window), each out to a pause up to a third of `d` further,
    a second short of the window's edges; with `longest` and a window longer, the first ends `longest` s after `s`
    and the second starts that before `e`, each in to a pause up to OVERLAP / 6 s. A known span (cached) in those
    ranges instead. `breaks`: the window's words', where they can be trusted (not a degenerate window's)."""
    if longest is not None and e - s > longest:
        h = OVERLAP / 6
        ends, starts = (s + longest - h, s + longest), (e - longest, e - longest + h)
        nominal = (s + longest, e - longest)
    else:
        mid = (s + e) / 2
        d = OVERLAP / 2 if e - s >= OVERLAP + 2.0 else (e - s) / 3
        ends = (mid + d, max(mid + d, min(mid + d * 4 / 3, e - 1.0)))
        starts = (min(mid - d, max(mid - d * 4 / 3, s + 1.0)), mid - d)
        nominal = (mid + d, mid - d)
    first = cuts.cached(known, (s, s), ends) or (s, cuts.pause_in(pauses, *ends, nominal[0], breaks))
    second = cuts.cached(known, starts, (e, e)) or (cuts.pause_in(pauses, *starts, nominal[1], breaks), e)
    return [first, second]


def reasked(units: list[dict], *, total: float, ask: Callable[[float, float], list[dict]], speech: SpeechSpans,
            voices: Voices, speakerless: list[tuple[float, float]],
            asked: Sequence[tuple[float, float]] = ()) -> tuple[list[dict], list[dict]]:
    """`--asr-repair quota`: up to REASK_CAP spans asked of 35t again (`ask(lo, hi)`: its words, absolute, times
    fixed): the misses with the most speech first, then repetition runs at talking speed, then the halves of
    speakerless windows. What a re-ask leaves wrong, or one not made, is Live's after it. Every span is cut at
    Silero pauses; `asked` holds the spans already asked (cached), taken where they fall in a cut's ranges."""
    plan: list[tuple[str, float, float, Any]] = []
    # one window at most
    spans = R.merged(R.gaps(units, total, speech.voiced), longest=WINDOW - 2 * REASK_PAD)
    runs = [r for r in R.runs(units) if not r.fast]
    pauses = cuts.pauses_of(speech.spans()) if spans or runs or speakerless else []
    breaks = cuts.word_breaks(units) if spans or runs or speakerless else []
    for a, b, found in sorted(spans, key=lambda m: -sum(g.voiced for g in m[2])):
        plan.append(("miss", *_padded(a, b, total, pauses, asked, breaks), found))
    plan += [("repeat", *_padded(r.start, r.end, total, pauses, asked, breaks), r) for r in runs]
    plan += [("speakerless", lo, hi, None) for s, e in speakerless
             for lo, hi in _split(s, e, pauses, asked, REASK_HALF, breaks)]
    log, done = [], []
    for kind, lo, hi, what in plan[:REASK_CAP]:
        entry = {"repair": kind, "via": "35t", "span": [round(lo, 2), round(hi, 2)]}
        log.append(entry)
        try:
            done.append((kind, lo, hi, what, ask(lo, hi), entry))
        except CacheMiss:
            entry["failed"] = "not cached (calls off)"
        except Exception as error:  # noqa: BLE001 -- given up (owner 2026-11-02); Live still tries it
            entry["failed"] = f"{type(error).__name__}: {error}"[:300]
    failed = [e for e in log if e.get("failed") and not e["failed"].startswith("not cached")]
    if failed:
        current_reporter().warning(
            "cloud-repair-failed", f"35t 重问 {len(failed)} 处失败（{failed[0]['failed']}）",
            impact="这几处改交 35t Live 补；metadata 里记了是哪几处")
    if len(plan) > REASK_CAP:
        log.append({"repair": "quota_cap", "left_to_live": len(plan) - REASK_CAP})
    units = [dict(u) for u in units]
    # Labels first, then runs from the last (their indices hold until a miss inserts), then misses.
    order = {"speakerless": 0, "repeat": 1, "miss": 2}
    done.sort(key=lambda d: (order[d[0]], -d[3].first if d[0] == "repeat" else 0))
    for kind, lo, hi, what, words, entry in done:
        mapping = R.relabel(words, voices)
        entry["speakers"] = mapping
        if kind == "speakerless":  # only the labels move: each loose word takes the re-ask word it overlaps most
            moved = 0
            for u in units:
                if not (u.get("nospk") and u.get("speaker") is None and lo <= u["start"] < hi):
                    continue
                best = max(words, key=lambda w: min(w["end"], u["end"]) - max(w["start"], u["start"]), default=None)
                if best is not None and min(best["end"], u["end"]) > max(best["start"], u["start"]):
                    u["speaker"] = mapping.get(best["speaker"])
                    moved += u["speaker"] is not None
            entry["labelled"] = moved
        elif kind == "repeat":
            run, speaker = what, units[what.first].get("speaker")
            got = [dict(w, speaker=mapping.get(w["speaker"], speaker)) for w in words
                   if w["start"] >= run.start - R.PAD and w["end"] <= run.end + R.PAD]
            longest = R.longest_run("".join(w["text"] for w in got), run.unit)
            entry.update(unit=run.unit, times=run.times, longest=longest)
            if got and longest < run.times / 2:
                units[run.first:run.stop] = got
                entry["replaced"] = len(got)
            else:
                entry["skipped"] = "kept"
        else:
            inserted = 0
            for g in what:
                inside = [dict(w, speaker=mapping.get(w["speaker"])) for w in words
                          if w["start"] >= g.start - R.PAD and w["end"] <= g.end + R.PAD]
                new = _trimmed(inside, units, g.start, g.end)
                at = R.insert_at(units, g.start)
                units[at:at] = new
                inserted += len(new)
            entry["inserted"] = inserted
    return units, log


def repaired(units: list[dict], *, total: float, language: str | None, live_cache: CallCache,
             live_wide_cache: CallCache, live, speech: SpeechSpans, voiceprints: Voiceprints,
             aligner: AlignerUse | None, reask=None, speakerless: list[tuple[float, float]] = (),
             asked: Sequence[tuple[float, float]] = ()) -> tuple[list[dict], list[dict]]:
    """The stitched units (output order) with what Live and local rules repair (`--asr-repair on`), and one log
    entry per spot; with `reask` (`quota`), 35t's own re-asks first (`reasked`, `asked` the spans it has cached).
    Words of a speakerless window carry `nospk`."""
    units = [dict(u) for u in units]
    log: list[dict] = []
    voices = Voices(speech.pcm, voiceprints, R.blocks([u for u in units if not u.get("nospk")]))
    if reask is not None:
        units, log = reasked(units, total=total, ask=reask, speech=speech, voices=voices,
                             speakerless=list(speakerless), asked=asked)

    # Speakerless windows: each block of their words takes the nearest voice (off until judged).
    loose = [u for u in units if u.get("nospk") and u.get("speaker") is None]
    if loose and not R.SPEAKERLESS_BY_VOICE:
        log.append({"repair": "speakerless", "via": "voice", "words": len(loose), "skipped": "off"})
    elif loose:
        found = R.blocks([dict(u, speaker="?") for u in loose])
        matched = 0
        for b in found:
            label = voices.nearest(b)
            if label is not None:
                matched += 1
                for u in loose:
                    if b["start"] <= u["start"] and u["end"] <= b["end"]:
                        u["speaker"] = label
        log.append({"repair": "speakerless", "via": "voice", "words": len(loose), "blocks": len(found),
                    "matched": matched, "labelled": sum(u["speaker"] is not None for u in loose)})

    # Spots for Live: misses, and repetition runs at talking speed; fast runs are trimmed here.
    found_runs = R.runs(units)
    spots = [_Spot("run", r.start, r.end, units[:r.first] + units[r.stop:], total, run=r)
             for r in found_runs if not r.fast]
    spots += [_Spot("gap", g.start, g.end, units, total, gap=g) for g in R.gaps(units, total, speech.voiced)]
    pcm = speech.pcm() if spots else None
    pauses = cuts.pauses_of(speech.spans()) if spots else []
    breaks = cuts.word_breaks(units) if spots else []
    # A wide window's edges, 35t's words', go out to the middle of a pause they lie in; a cached window between
    # those and its old edges (named so before 2026-11-22) is taken (`cuts.widened`).
    wide_known = cuts.spans_named(live_wide_cache.names(), "ctx") if spots else []
    for spot in spots:
        spot.lo, spot.hi = cuts.widened(spot.lo, spot.hi, pauses, 0.0, total, wide_known)
    align = None
    if aligner and spots:
        from .aligner import aligned_units

        broken: list[str] = []

        def align(segment: dict, neighbours: list[dict]) -> list[dict] | None:
            if broken:  # failed once (no model, no transformers): not again for every spot
                return None
            try:
                return aligned_units([segment], all_segments=[segment, *neighbours], pcm16k=pcm,
                                     device=aligner.device).get(0)
            except Exception as error:  # noqa: BLE001 -- S5b, as for a segment the aligner gives up on
                broken.append(str(error))
                current_reporter().warning("cloud-repair-aligner", f"强制对齐失败（{error}），修补补回的字改用 Silero 定时")
                return None

    def listen(a: float, b: float) -> Callable[[], dict]:
        return lambda: {"text": live(pcm[int(a * live_client.SR):int(b * live_client.SR)].tobytes(), language=language)}

    def wide(spot: _Spot) -> None:
        try:
            spot.text = live_wide_cache.fetch(f"ctx-{spot.lo:.1f}-{spot.hi:.1f}", listen(spot.lo, spot.hi))["text"]
            spot.heard = "wide"
        except Exception as error:  # noqa: BLE001 -- the spot alone, a shorter session, may still come back; and
            spot.wide_error = str(error)[:300]  # a replay (CacheMiss) reads what the spot alone answered then
        else:
            spot.wide_unused = spot.deaf()

    def alone(spot: _Spot) -> None:
        try:
            text = live_cache.fetch(spot.name, listen(*cuts.padded(spot.start, spot.end, LIVE_PAD, pauses, total,
                                                                   breaks=breaks)))["text"]
        except CacheMiss:
            error = "not cached (calls off)"
        except Exception as e:  # noqa: BLE001 -- a failed spot is given up, the task goes on (owner 2026-11-02)
            error = str(e)[:300]
        else:
            spot.text, spot.heard = text, "alone"
            return
        if spot.text is None:
            spot.error = error
        else:  # the wide answer stands, and is judged as it is
            spot.alone_error = error

    if spots:
        from concurrent.futures import ThreadPoolExecutor

        # The workers report where this thread does (the upload notice is said through the first Live call).
        with ThreadPoolExecutor(live_client.SESSIONS, initializer=bind_reporter, initargs=(current_reporter(),)) as pool:
            list(pool.map(wide, spots))
            list(pool.map(alone, [s for s in spots if s.wide_error is not None or s.wide_unused is not None]))

    # Runs, last first so earlier indices hold.
    by_run = {id(s.run): s for s in spots if s.run is not None}
    for r in sorted(found_runs, key=lambda r: -r.first):
        if r.fast:
            patch = R.patch_fast(units, r)
            if patch is not None:
                units[r.first:r.stop] = patch
            log.append({"repair": "repeat", "via": "local", "span": [round(r.start, 2), round(r.end, 2)],
                        "unit": r.unit, "times": r.times, "kept": len(patch) // r.size if patch else r.times})
            continue
        spot = by_run[id(r)]
        if spot.text is None:
            log.append(spot.entry(unit=r.unit, times=r.times, failed=spot.error))
            continue
        text, why, script = spot.new_text()
        longest = R.longest_run(text, r.unit)
        rest = units[:r.first] + units[r.stop:]
        why = (why or R.implausible(text, speech.voiced(r.start, r.end), language, script=script)
               or (None if longest < r.times / 2 else "kept"))
        entry = spot.entry(unit=r.unit, times=r.times, longest=longest, text=text[:80])
        if why is None:
            timed, entry["timing"] = _timed(text, r.start, r.end, units[r.first].get("speaker"),
                                            _voiced_spans(pcm, r.start, r.end), _near(rest, r.start, r.end), align)
            new = _trimmed(timed, rest, r.start, r.end)
            why = spot.unused(new)
            if why is None:
                units[r.first:r.stop] = new
                entry["replaced"] = len(new)
        if why is not None:
            entry["skipped"] = why
        log.append(entry)

    # Misses: Live's text right after the unit that ends before the gap (no global sort); it joins that
    # unit's turn when the voice gives it the same speaker.
    for spot in sorted((s for s in spots if s.gap is not None), key=lambda s: -s.start):
        g = spot.gap
        if spot.text is None:
            log.append(spot.entry(failed=spot.error))
            continue
        text, why, script = spot.new_text()
        entry = spot.entry(voiced=round(g.voiced, 1), text=text[:80])
        why = why or R.implausible(text, g.voiced, language, script=script)
        new = []
        if why is None:
            heard = _voiced_spans(pcm, g.start, g.end)
            speaker = voices.nearest({"start": g.start, "end": g.end, "voiced": heard})
            timed, entry["timing"] = _timed(text, g.start, g.end, speaker, heard, _near(units, g.start, g.end), align)
            new = _trimmed(timed, units, g.start, g.end)
            entry["speaker"] = speaker
            why = spot.unused(new, MIN_NEW)
        if why is None:
            at = R.insert_at(units, g.start)
            units[at:at] = new
            entry["inserted"] = len(new)
        else:
            entry["skipped"] = why
        log.append(entry)
    # given up on a failed call: no answer at all, or a wide one that did not hear the spot and the spot alone failed
    failed = [e for s in spots for e in (s.error if s.text is None else s.alone_error,)
              if e and not e.startswith("not cached")]
    if failed:
        current_reporter().warning(
            "cloud-repair-failed", f"35t Live 补 {len(failed)} 处失败（{failed[0]}）",
            impact="这几处照 35t 原来的写法留着；metadata 里记了是哪几处")
    return units, log


def _near(units: list[dict], a: float, b: float) -> list[dict]:
    """The units just either side of [a, b], which bound the aligner's audio."""
    before = [u for u in units if u["end"] <= a + R.PAD]
    after = [u for u in units if u["start"] >= b - R.PAD]
    return ([max(before, key=lambda u: u["end"])] if before else []) + ([min(after, key=lambda u: u["start"])] if after else [])


def _continued(last: float, floor: float, pauses: cuts.Pauses,
               breaks: cuts.Breaks = ()) -> tuple[tuple[float, float], float]:
    """Where a window that stopped early goes on (the range a cached window may start in, and the start): at a pause
    CONTINUE_BACK s before its `last` word, not before `floor`; when the floor is the later, at a pause up to
    CONTINUE_BACK[1] - CONTINUE_BACK[0] s after it. `breaks`: the window's words'."""
    width = CONTINUE_BACK[1] - CONTINUE_BACK[0]
    nominal = max(last - CONTINUE_BACK[0], floor)
    lo = max(last - CONTINUE_BACK[1], floor)
    hi = max(last - CONTINUE_BACK[0], floor + width)
    return (lo, hi), cuts.pause_in(pauses, lo, hi, nominal, breaks)


def _trimmed(new: list[dict], around: list[dict], a: float, b: float) -> list[dict]:
    left, right = R.neighbours(around, a, b)
    first, end = R.trim_echo([u["text"] for u in new], left, right)
    return new[first:end]


# --- one source -------------------------------------------------------------------------------------------


def _window_read(raw: dict, start: float, end: float) -> dict:
    """One window's answer read: its words (times fixed, `fix_times`), status, longest repeat, whether it gave no
    speakers."""
    words, events = fix_times(raw_words(raw, start))
    rep, unit = longest_repeat(words)
    no_speaker = bool(words) and not any(w["speaker"] for w in words)
    if no_speaker:  # for the repair (`nospk`); `Unit` does not keep it
        words = [dict(x, nospk=True) for x in words]
    return {"start": start, "end": end, "status": answer_status(raw), "words": words, "events": events,
            "repeat": rep, "repeat_unit": unit, "no_speaker": no_speaker}


def identity(source: Path, language: str | None) -> dict:
    return {"provider": GEMINI_35T, "model": MODEL, "window": WINDOW, "overlap": OVERLAP,
            "language": language, "audio": audio_fingerprint(source)}


def transcribe(
    source: str | Path,
    *,
    cache_root: str | Path,
    language: str | None,
    repair: str,
    offline: bool = False,
    caller: Caller | None = None,
    duration: float | None = None,
    live=None,
    speech: SpeechSpans | None = None,
    voiceprints: Voiceprints | None = None,
    aligner: AlignerUse | None = None,
    answer: dict | None = None,
) -> Transcript:
    """The source's 35t transcript; every call cached under `cache_root/gemini-35t/`, Live's under
    `cache_root/gemini-35t-live-ctx/` and `cache_root/gemini-35t-live/`. `repair` is `--asr-repair` (`repair.MODES`).
    `answer`: one answer over the whole source from 0 s (a transcript file, `--asr-transcript`), read as a window's
    (its times fixed): no cache, no call, no split of a degenerate window, no repair (those would call)."""

    src = Path(source)
    code = LANGUAGE_CODES.get((language or "").lower())
    cache = CallCache(cache_root, GEMINI_35T, identity=identity(src, code), offline=offline)
    call = caller or Caller()
    total = duration if duration is not None else probe_media_duration(src)

    read: set[str] = set()  # the windows read so far, by name: a later step never takes one of them again

    def window(start: float, end: float) -> dict:
        def ask() -> dict:
            current_reporter().debug("gemini-35t call", {"start": round(start, 1), "end": round(end, 1)})
            with tempfile.TemporaryDirectory(prefix="finesub-35t-") as tmp:
                clip = extract_upload_mp3(src, start, end, Path(tmp) / "window.mp3")
                return call(clip, language=code, seconds=end - start)

        read.add(f"w-{start:.1f}-{end:.1f}")
        return _window_read(cache.fetch(f"w-{start:.1f}-{end:.1f}", ask), start, end)

    if answer is not None:
        part = _window_read(answer, 0.0, total)
        log = [dict(e, window=[0.0, total]) for e in part["events"]]
        if part["repeat"] >= REPEAT:
            log.append({"repeat": part["repeat"], "unit": part["repeat_unit"]})
        if part["no_speaker"]:
            log.append({"no_speaker_windows": 1})
        return Transcript(GEMINI_35T, MODEL, total, "word", turns_from_units(Unit.from_dict(u) for u in part["words"]),
                          log)

    speech = speech or SpeechSpans(src)  # decoded on first use: by the cuts and the repair, once
    silences: list[tuple[float, float]] | None = None

    def pauses() -> list[tuple[float, float]]:  # only a source cut anywhere needs Silero
        nonlocal silences
        if silences is None:
            silences = cuts.pauses_of(speech.spans())
        return silences

    named = cuts.spans_named(cache.names(), "w")

    def known() -> list[tuple[float, float]]:
        """The cached windows not read yet (a tail goes on from a degenerate window's half: not that half again)."""
        return [k for k in named if f"w-{k[0]:.1f}-{k[1]:.1f}" not in read]

    def span(starts: tuple[float, float], start: float) -> tuple[tuple[float, float], bool, bool]:
        """The window from `start` (or a cached one starting in `starts`), whether it is the last, and whether it is
        a window already read (its answer again)."""
        ends = lambda s: (total, total) if total - s <= WINDOW else (s + WINDOW - CACHED_REACH, s + WINDOW)  # noqa: E731
        found = cuts.cached(known(), starts, ends)
        if found is not None:  # named to 0.1 s
            return found, found[1] >= total - cuts.NAMED, False
        planned = (start, total) if total - start <= WINDOW else (start, cuts.end_before(start + WINDOW, pauses(), REACH))
        return planned, planned[1] == total, f"w-{planned[0]:.1f}-{planned[1]:.1f}" in read

    parts, log, split = [], [], set()
    nxt, starts, tail_done = 0.0, (0.0, 0.0), False
    while True:
        (start, end), final, again = span(starts, nxt)
        w = window(start, end)
        degenerate = w["repeat"] >= REPEAT or w["status"] != "completed"
        # Only a tail can plan a window already read (a half of a split one): its words are in already, but a
        # degenerate one not split yet is split now, its halves in its place.
        if again and (not degenerate or (start, end) in split):
            log.append({"tail_read_already": [round(start, 1), round(end, 1)]})
            break
        if again:
            name = f"w-{start:.1f}-{end:.1f}"
            parts = [p for p in parts if f"w-{p['start']:.1f}-{p['end']:.1f}" != name]
        if degenerate:
            split.add((start, end))
            halves = [window(a, b) for a, b in _split(start, end, pauses(), known())]
            log.append({"split": [start, end], "repeat": w["repeat"], "unit": w["repeat_unit"], "status": w["status"]})
            parts.extend(halves)
            last = max((x["end"] for h in halves for x in h["words"]), default=start)
            breaks = cuts.word_breaks(halves[-1]["words"])  # the whole window's words are not trusted
        else:
            parts.append(w)
            last = max((x["end"] for x in w["words"]), default=start)
            breaks = cuts.word_breaks(w["words"])
        if final:
            if total - last > GAP and not tail_done:
                log.append({"tail": [last, total]})
                starts, nxt = _continued(last, start + 1.0, pauses(), breaks)
                tail_done = True
                continue
            break
        if end - last > GAP:
            starts, nxt = _continued(last, start + MIN_ADVANCE, pauses(), breaks)
            log.append({"early_stop": [start, end], "last_word": round(last, 1), "next": round(nxt, 1)})
        else:
            starts, nxt = (end - cuts.OVERLAP[1], end - cuts.OVERLAP[0]), cuts.start_before(end, pauses(), breaks=breaks)
    units = stitch([(p["start"], p["end"], p["words"]) for p in parts], segments=False)
    log.extend(dict(e, window=[p["start"], p["end"]]) for p in parts for e in p["events"])
    if any(p["no_speaker"] for p in parts):
        log.append({"no_speaker_windows": sum(p["no_speaker"] for p in parts)})
    mode = R.mode(repair)
    if mode != "off":
        asked = {"model": live_client.MODEL, "fed": "mix", "language": language, "audio": audio_fingerprint(src)}
        live_cache = CallCache(cache_root, LIVE_CACHE, offline=offline,
                               identity={"provider": LIVE_CACHE, "pad": LIVE_PAD, **asked})
        live_wide_cache = CallCache(cache_root, LIVE_WIDE_CACHE, offline=offline,
                                    identity={"provider": LIVE_WIDE_CACHE, **asked})
        reask, asked = None, []
        if mode == "quota":
            reask_cache = CallCache(cache_root, REASK_CACHE, offline=offline,
                                    identity=dict(identity(src, code), provider=REASK_CACHE, pad=REASK_PAD))

            def reask(lo: float, hi: float) -> list[dict]:
                def ask() -> dict:
                    current_reporter().debug("gemini-35t repair call", {"start": round(lo, 1), "end": round(hi, 1)})
                    with tempfile.TemporaryDirectory(prefix="finesub-35t-") as tmp:
                        clip = extract_upload_mp3(src, lo, hi, Path(tmp) / "span.mp3")
                        return call(clip, language=code, seconds=hi - lo)

                return fix_times(raw_words(reask_cache.fetch(f"r-{lo:.1f}-{hi:.1f}", ask), lo))[0]

            asked = cuts.spans_named(reask_cache.names(), "r")

        units, fixes = repaired(units, total=total, language=language, live_cache=live_cache,
                                live_wide_cache=live_wide_cache, live=live or live_client.Live(), speech=speech,
                                voiceprints=voiceprints or VOICEPRINTS, aligner=aligner, reask=reask,
                                speakerless=[(p["start"], p["end"]) for p in parts if p["no_speaker"]],
                                asked=asked)
        log.extend(fixes)
    return Transcript(GEMINI_35T, MODEL, total, "word", turns_from_units(Unit.from_dict(u) for u in units), log)
