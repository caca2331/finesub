"""ElevenLabs Scribe v3.

One multipart call per source, the source as 16 kHz mono FLAC: Scribe takes up to 15 h and 5 GB a file (15 h of this
FLAC is about 1 GB) and transcribes anything over 8 minutes in up to four parallel parts itself, which is trusted
(owner 2026-11-21: 「可以相信服务器上的切分的处理已经是他们优化过了的了」). Only a source longer than LIMIT is cut
here, at Silero pauses (`cuts.window`; a piece's start where the previous piece's words break too) into pieces of at
most LONGEST: each piece is one call (`p-<start>-<end>`), stitched as MAI2's windows are (each piece's labels kept
apart, `stitch`); each piece is repaired on its own words before the stitch, so a repair never replaces another
piece's. The answer times every word -- in Japanese and Chinese every character -- and labels it with a speaker; `spacing` tokens are dropped (the line builder joins words by language)
and so are `audio_event` tokens (laughter, applause), counted in the log. Keys: the `ELEVENLABS` pool in `.env`, front to back
(`scribe_keys`; a key needs the speech_to_text permission, and user_read to have its balance checked).

What the adapter fixes in the answer: a stretch of talk returned as one word token is split into words of its
own characters (`unglued`); a mark's time is not the word's (`_closing_marks`), a bracketed note for what Scribe
could not make out is dropped (`without_notes`; both in `answer_units`). Repair (`--asr-repair on`,
`docs/joint-asr.md`「退化的修补」): a timestamp collapse (`collapses`) and a glued word are asked of Scribe again for
their span only (`reask`), REASK_PAD s of audio either side out to a Silero pause (`cuts.padded`); the span's words are replaced by what the re-ask hears
between the words either side, its speakers matched by voice as MAI2's re-asks are (a glued word's label matched
to no one but with RESCUE_MIN s of voice is a new person, `c<k>:`). A glued word not re-asked keeps the split
and its one speaker label, with a warning.
"""

from __future__ import annotations

import json
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from finesub.api_http import CONNECT_TIMEOUT_SECONDS, RETRYABLE_TRANSPORT, api_http_client, retryable_status
from finesub.media.ffmpeg import _clip_duration, _format_seconds, _run_ffmpeg, probe_media_duration, resolve_ffmpeg
from finesub.reporting import current_reporter

from . import SCRIBE, cuts
from . import repair as R
from .cache import CacheMiss, CallCache, audio_fingerprint
from .model import Transcript, Unit, turns_from_units
from .mosspro import SpeechSpans
from .relink import VOICEPRINTS, Voiceprints, Voices
from .stitch import stitch

MODEL = "scribe_v3"
API = "https://api.elevenlabs.io/v1/speech-to-text"
KEY_ENV = "ELEVENLABS"
#: The single call's cache name: the whole source.
CALL = "full"
#: The longest file ElevenLabs takes (docs, 2026-11-21): a source up to it is sent whole.
LIMIT = 15 * 3600.0
#: A longer one is cut into pieces of at most this: a minute under LIMIT.
LONGEST = LIMIT - 60.0
#: The cache directory of the repair's re-asks, beside the call's.
REPAIR_CACHE = f"{SCRIBE}-repair"
REASK_PAD = 10.0
#: How far a re-asked word's start may be off the same word in the first answer (hina-73: up to 0.08 s).
EDGE_SLACK = 0.1
#: Scribe's own segmentation (`additional_formats`), kept in the cached answer for comparison only.
SEGMENT_EXPORT = {"format": "segmented_json", "segment_on_silence_longer_than_s": 0.5, "max_segment_duration_s": 7}
RETRY_WAITS = (30.0, 60.0, 120.0)
MAX_RETRY_AFTER = 300.0
#: Lines start this much earlier (`lines.widened`): against the human word times, Scribe's line starts are 0.17 s
#: later than production's (median); its ends are not late (2026-11-11 report, arm E1st).
START_PAD = 0.17


#: Scribe v3's price per audio hour, pay-as-you-go (ElevenLabs' price list, 2026-11-11), for the estimate it prints.
USD_PER_HOUR = 0.20


class ScribeError(RuntimeError):
    pass


class ScribeRefused(ScribeError):
    """The key was refused (401, 402, 403): bad, without the permission, or out of credits -- the answer does not
    say which reliably, so the pool moves on rather than reads it."""

    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:2000]}")
        self.status = status


#: Scribe writes a stretch it could not make out as ordinary words in brackets (「（聞き取り不能）」, character by
#: character in Japanese): a bracketed run of at most this many words is dropped, brackets included.
NOTE_MAX_WORDS = 12
#: Square brackets too: miyako's glued word opens on an unpaired 「[聞き取り不能）」.
_OPEN, _CLOSE = "（([", "）)]"


def without_notes(words: list[dict]) -> tuple[list[dict], int]:
    """(`words` minus the bracketed notes, how many notes were dropped)."""
    out, notes, i = [], 0, 0
    while i < len(words):
        text = (words[i].get("text") or "").strip()
        if words[i].get("type") == "word" and text[:1] in _OPEN:
            close = next((j for j in range(i, min(len(words), i + NOTE_MAX_WORDS + 2))
                          if any(ch in _CLOSE for ch in words[j].get("text") or "")), None)
            if close is not None:
                notes += 1
                last = words[close]["text"]
                rest = last[max(last.rfind(ch) for ch in _CLOSE) + 1:].strip()  # 「）。」 keeps its 「。」
                if rest:
                    out.append({**words[close], "text": rest})
                i = close + 1
                continue
        out.append(words[i])
        i += 1
    return out, notes


#: A timestamp collapse: at least COLLAPSE_WORDS words in a row (spacing aside) with zero duration, strung over at
#: least COLLAPSE_SPAN s. Scribe moved hina-73's 514-529 s this way to 559-593 s; a burst said at one point
#: (「え？」 seven times, 「哎呀」 over and over) is zero-length too but strung over a few seconds at most. The words
#: of the run's speaker starting within COLLAPSE_TAIL s of its last belong to it: hina-73's run ends on a 40 ms 「れ」.
COLLAPSE_WORDS = 15
COLLAPSE_SPAN = 15.0
COLLAPSE_TAIL = 0.3


def collapses(words: list[dict]) -> list[tuple[float, float]]:
    """(first start, last start) of every timestamp collapse in the answer's `words`."""
    words = [w for w in words if w.get("type") == "word" and w.get("start") is not None and w.get("end") is not None]
    out, i = [], 0
    while i < len(words):
        j = i
        while j < len(words) and float(words[j]["end"]) <= float(words[j]["start"]):
            j += 1
        first, last = float(words[i]["start"]), float(words[j - 1]["start"]) if j > i else 0.0
        if j - i >= COLLAPSE_WORDS and last - first >= COLLAPSE_SPAN:
            speaker = words[j - 1].get("speaker_id")
            while (j < len(words) and words[j].get("speaker_id") == speaker
                   and float(words[j]["start"]) <= last + COLLAPSE_TAIL):
                j += 1
            out.append((first, float(words[j - 1]["start"])))
        i = max(j, i + 1)
    return out


#: A glued word: Scribe once returned 983 s of talk (miyako, 4263 characters, from an unpaired 「[」 to the next 「]」)
#: as one word token -- its characters timed as usual, one speaker label on all of it, and to the line DP one unit
#: it cannot cut. A word token is glued when it holds GLUE_LETTERS letters and digits or more and is not one unit of
#: at most REPEAT_UNIT of them said over and over (REPEAT_SHARE of it in one run: `SPRITE` x31 in one 9 s token is a
#: repetition, left as it was), or when it holds GLUE_SENTENCES sentence ends with letters either side. Across 37
#: answers (11.8 h, 177 001 word tokens) an ordinary word has at most 16 and none a sentence end between letters;
#: duration tells nothing (an ordinary word spans up to 20.6 s).
GLUE_LETTERS = 40
GLUE_SENTENCES = 2
REPEAT_UNIT = 10
REPEAT_SHARE = 0.8
_SENTENCE_ENDS = set("。？！?!")


def is_glued(word: dict) -> bool:
    if word.get("type") != "word":
        return False
    text = word.get("text") or ""
    letters = R.key(text)
    if len(letters) >= GLUE_LETTERS and not any(
            R.longest_run(text, letters[:n]) * n >= REPEAT_SHARE * len(letters) for n in range(1, REPEAT_UNIT + 1)):
        return True
    at = [i for i, ch in enumerate(text) if R.key(ch)]
    return bool(at) and sum(ch in _SENTENCE_ENDS for ch in text[at[0]:at[-1]]) >= GLUE_SENTENCES


def _wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in "WF"


def split_glued(word: dict) -> list[dict]:
    """A glued word as words of its own characters, shaped as Scribe writes them: a CJK or kana character is a word,
    a run of narrow letters and digits with the marks inside it (`Stratford-on-Avon`, `HP0`) is one, other marks go
    on the front of the next word (「。光」) and the last ones on the last word; a space ends a word. Empty when its
    characters are not all timed."""
    chars = word.get("characters") or []
    if not chars or any(c.get("start") is None or c.get("end") is None for c in chars):
        return []
    words: list[list[dict]] = []
    cur: list[dict] = []
    pending: list[dict] = []
    for c in chars:
        ch = (c.get("text") or "")[:1]
        narrow_tail = bool(cur) and not pending and not _wide(cur[-1]["text"][-1:] or " ")
        if not ch.strip():
            cur = []
        elif unicodedata.category(ch)[0] in "LN":
            if narrow_tail and not _wide(ch):
                cur.append(c)
            else:
                cur = [*pending, c]
                pending = []
                words.append(cur)
        elif narrow_tail and not _wide(ch) or unicodedata.category(ch)[0] == "M" and cur and not pending:
            cur.append(c)
        else:
            cur = []
            pending.append(c)
    if pending:
        if words:
            words[-1].extend(pending)
        else:
            words.append(pending)
    return [{"type": "word", "text": "".join(c["text"] for c in cs), "start": cs[0]["start"], "end": cs[-1]["end"],
             "speaker_id": word.get("speaker_id"), "characters": cs} for cs in words]


@dataclass(frozen=True)
class Glued:
    """A glued word: its first and last character's start, its length."""

    first: float
    last: float
    chars: int


def unglued(words: list[dict]) -> tuple[list[dict], list[Glued]]:
    """(`words` with each glued word split (`split_glued`; one whose characters are not timed stays whole), the
    glued words)."""
    out, found = [], []
    for w in words:
        if not is_glued(w):
            out.append(w)
            continue
        parts = split_glued(w)
        starts = [float(c["start"]) for c in w.get("characters") or [] if c.get("start") is not None]
        found.append(Glued(min(starts, default=float(w["start"])), max(starts, default=float(w["start"])),
                           len(w.get("text") or "")))
        out += parts or [w]
    return out, found


def shifted(words: list[dict], offset: float) -> list[dict]:
    """An answer's `words` (and their characters) moved `offset` s on: a piece's times as the source's."""

    def moved(x: dict) -> dict:
        return {**x, **{k: float(x[k]) + offset for k in ("start", "end") if x.get(k) is not None}}

    return [{**moved(w), "characters": [moved(c) for c in w["characters"]]} if w.get("characters") else moved(w)
            for w in words]


def answer_units(words: list[dict], offset: float = 0.0) -> tuple[list[dict], int]:
    """(word units in output order, how many audio_event tokens were dropped) of an answer's `words`; bracketed
    notes dropped too (`without_notes`). An entry without `type` is a word: Scribe always sends one, a transcript
    file written by hand in its shape may not."""
    units, events = [], 0
    words, _ = without_notes(words)
    for w in words:
        kind = w.get("type", "word")
        if kind == "audio_event":
            events += 1
            continue
        if kind != "word" or w.get("start") is None or w.get("end") is None or not (w.get("text") or "").strip():
            continue
        text, start, end = w["text"].strip(), float(w["start"]), float(w["end"])
        # A mark's character can sit on the far side of a pause ("What？" spans 11 s, its 「？」 at the end;
        # 「。26」 starts on the 「。」 before the pause): a word's span is its other characters'.
        core = [c for c in w.get("characters") or [] if c.get("start") is not None and c.get("end") is not None
                and (c.get("text") or "").strip() and _leading_marks(c["text"]) == 0]
        if core:
            start, end = float(core[0]["start"]), max(float(core[-1]["end"]), float(core[-1]["start"]))
        lead = _closing_marks(text)
        if 0 < lead < len(text):  # a closing mark goes to the word before (the DP folds it in); an opening one stays
            units.append({"start": round(float(w["start"]) + offset, 3), "end": round(float(w["start"]) + offset, 3),
                          "text": text[:lead], "speaker": w.get("speaker_id")})
            text = text[lead:]
        units.append({"start": round(start + offset, 3), "end": round(end + offset, 3), "text": text,
                      "speaker": w.get("speaker_id")})
    return units, events


#: Marks that close what came before: a word starting with them gives them to the word before.
_CLOSING = set("。，、？！；：.,?!;:…")


def _closing_marks(text: str) -> int:
    """How many closing marks (sentence and clause ends, closing brackets and quotes) `text` starts with."""
    n = 0
    while n < len(text) and (text[n] in _CLOSING or unicodedata.category(text[n]) in ("Pe", "Pf")):
        n += 1
    return n


def _leading_marks(text: str) -> int:
    """How many punctuation characters `text` starts with."""
    n = 0
    while n < len(text) and unicodedata.category(text[n])[0] == "P":
        n += 1
    return n


def form(language: str | None, seed: int) -> dict:
    data = {"model_id": MODEL, "diarize": "true", "timestamps_granularity": "character", "tag_audio_events": "true",
            "seed": str(seed), "additional_formats": json.dumps([SEGMENT_EXPORT])}
    if language:
        data["language_code"] = language.split("-")[0]
    return data


def upload_flac(source: Path, out: Path, span: tuple[float, float] | None = None) -> Path:
    """The source (or its [start, end) `span`) as Scribe is sent it: 16 kHz mono FLAC."""
    cut = ["-ss", _format_seconds(span[0]), "-t", _format_seconds(_clip_duration(*span))] if span else []
    _run_ffmpeg([resolve_ffmpeg(), "-y", "-nostdin", "-loglevel", "error", *cut, "-i", str(source),
                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(out)])
    return out


@dataclass(frozen=True)
class Caller:
    key: str
    seed: int = 20261111
    sleep: Callable[[float], None] = time.sleep

    def __call__(self, audio: Path, *, language: str | None) -> dict:
        import httpx

        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                with api_http_client(httpx.Timeout(1800.0, connect=CONNECT_TIMEOUT_SECONDS)) as client, \
                        audio.open("rb") as f:
                    r = client.post(API, headers={"xi-api-key": self.key}, data=form(language, self.seed),
                                    files={"file": (audio.name, f, "audio/flac")})
            except RETRYABLE_TRANSPORT:
                if attempt == len(RETRY_WAITS):
                    raise
                self.sleep(RETRY_WAITS[attempt])
                continue
            if r.status_code < 400:
                return r.json()
            if retryable_status(r.status_code) and attempt < len(RETRY_WAITS):
                after = r.headers.get("Retry-After", "")
                wait = float(after) if after.replace(".", "", 1).isdigit() else RETRY_WAITS[attempt]
                if wait > MAX_RETRY_AFTER:
                    raise ScribeError(f"HTTP {r.status_code}: the service asks to wait {wait:.0f} s")
                self.sleep(wait)
                continue
            if r.status_code in (401, 402, 403):
                raise ScribeRefused(r.status_code, r.text)
            raise ScribeError(f"HTTP {r.status_code}: {r.text[:2000]}")
        raise ScribeError("no answer after retries")


def identity(source: Path, language: str | None) -> dict:
    return {"provider": SCRIBE, "model": MODEL, "language": language, "audio": audio_fingerprint(source)}


def transcribe(source: str | Path, *, cache_root: str | Path, language: str | None, repair: str,
               caller: Callable[..., dict] | None = None, offline: bool = False, duration: float | None = None,
               speech: SpeechSpans | None = None, voiceprints: Voiceprints | None = None,
               answer: dict | None = None) -> Transcript:
    """The source's Scribe transcript, words in output order; the call cached under `cache_root/scribe/`, the
    repair's under `cache_root/scribe-repair/`. `repair` is `--asr-repair` (`repair.MODES`; `quota` is `on`).
    `answer`: one answer over the whole source from 0 s (a transcript file, `--asr-transcript`), read as the whole-
    source call's: no cache, no call, no repair (that would call)."""
    if answer is not None:
        repair = "off"
    src = Path(source)
    cache = CallCache(cache_root, SCRIBE, identity=identity(src, language), offline=offline)
    total = duration if duration is not None else probe_media_duration(src)

    def call(span: tuple[float, float] | None, whole: bool) -> dict:
        """Scribe's answer for `span` (None: the whole source); `whole` for a main call, whose cost the pool says."""
        if caller is None:
            raise ScribeError(f"--asr scribe needs {KEY_ENV} in .env")
        lo, hi = span or (0.0, total)
        current_reporter().debug("scribe call", {"span": [round(lo, 1), round(hi, 1)]})
        with tempfile.TemporaryDirectory(prefix="finesub-scribe-") as tmp:
            return caller(upload_flac(src, Path(tmp) / "audio.flac", span), language=language, seconds=hi - lo,
                          whole=whole)

    log: list[dict] = []
    repair_cache = None

    def repaired(units: list[dict], found: list[tuple[float, float]], glued: list[Glued]) -> list[dict]:
        """`units` with their collapses and glued words asked again (`reask`), or as they are, logged."""
        nonlocal repair_cache, speech
        if not (found or glued) or R.mode(repair) == "off":
            if found:
                log.append({"collapses": [[round(a, 2), round(b, 2)] for a, b in found]})
            log.extend({"glued": [round(g.first, 2), round(g.last, 2)], "chars": g.chars, "via": "split"} for g in glued)
            return units
        if repair_cache is None:
            repair_cache = CallCache(cache_root, REPAIR_CACHE, offline=offline,
                                     identity=dict(identity(src, language), provider=REPAIR_CACHE, pad=REASK_PAD))
        speech = speech or SpeechSpans(src)

        # Named to 0.1 s: a detection change that moves `lo` by less reuses the answer, offset by the new `lo`.
        def ask(lo: float, hi: float) -> list[dict]:
            got = repair_cache.fetch(f"r-{lo:.1f}-{hi:.1f}", lambda: call((lo, hi), False)).get("words") or []
            if any(is_glued(w) for w in got):
                raise ScribeError("the re-ask glued a stretch of talk into one word again")
            return answer_units(got, lo)[0]

        units, fixes = reask(units, found, glued, total=total, ask=ask, speech=speech,
                             voiceprints=voiceprints or VOICEPRINTS, known=cuts.spans_named(repair_cache.names(), "r"))
        log.extend(fixes)
        return units

    if total <= LIMIT or answer is not None:
        raw = answer if answer is not None else cache.fetch(CALL, lambda: call(None, True))
        words, glued = unglued(raw.get("words") or [])
        units, events = answer_units(words)
        if events:
            log.append({"audio_events_dropped": events})
        units = repaired(units, collapses(words), glued)
    else:
        speech = speech or SpeechSpans(src)
        pauses, known = cuts.pauses_of(speech.spans()), cuts.spans_named(cache.names(), "p")
        read, events, after, breaks = [], 0, None, []
        while True:  # every main call before any repair, each answer kept as units and defects only
            (lo, hi), last = cuts.window(0.0, total, LONGEST, pauses, after, breaks=breaks, known=known,
                                             cached_reach=cuts.CACHED_REACH)
            raw = cache.fetch(f"p-{lo:.1f}-{hi:.1f}", lambda: call((lo, hi), True))
            words, glued = unglued(shifted(raw.get("words") or [], lo))
            got, dropped = answer_units(words)
            events += dropped
            read.append((lo, hi, got, collapses(words), glued))
            del raw, words
            if last:
                break
            after, breaks = hi, cuts.word_breaks(got)  # the next piece starts where this one's words break too
        log.append({"pieces": [[round(lo, 1), round(hi, 1)] for lo, hi, *_ in read]})
        if events:
            log.append({"audio_events_dropped": events})
        # repaired on its own words: a repair across a seam replaces none of the other piece's
        units = stitch([(lo, hi, repaired(got, found, glued)) for lo, hi, got, found, glued in read], segments=False)
    _warn_split(log)
    return Transcript(SCRIBE, MODEL, total, "word", turns_from_units(Unit.from_dict(u) for u in units), log)


def _warn_split(log: list[dict]) -> None:
    """One warning for the glued words left split (`split_glued`): their speakers are one label."""
    split = [e for e in log if "glued" in e and e.get("via") == "split"]
    if not split:
        return
    where = "、".join(f"{e['glued'][0]:.0f}–{e['glued'][1]:.0f} s（{e['chars']} 字）" for e in split)
    why = f"重转失败：{split[0]['failed']}" if split[0].get("failed") else "--asr-repair off，没有重转"
    current_reporter().warning(
        "scribe-glued", f"Scribe 把 {len(split)} 段话粘成了一个词（{where}），按逐字时间拆开了（{why}）",
        impact="这几段的说话人只是原来那一个标签（整段算成同一个人），不可信；metadata 的 log 里记了是哪几段")


# --- repair -----------------------------------------------------------------------------------------------


def _spans(found: list[tuple[float, float]], glued: list[Glued]) -> list[tuple[float, float, list[Glued]]]:
    """The collapses and glued words as (first, last, the glued words in it), by start; spots closer than
    `repair.MERGE_WITHIN` are one span."""
    spots = sorted([(a, b, []) for a, b in found] + [(g.first, g.last, [g]) for g in glued], key=lambda s: s[0])
    out: list[tuple[float, float, list[Glued]]] = []
    for first, last, gs in spots:
        if out and first - out[-1][1] < R.MERGE_WITHIN:
            out[-1] = (out[-1][0], max(out[-1][1], last), out[-1][2] + gs)
        else:
            out.append((first, last, gs))
    return out


def _entry(first: float, last: float, glued: list[Glued], lo: float, hi: float) -> dict:
    kind = "glued" if glued else "collapse"
    entry = {"repair": kind, "via": "scribe", "span": [round(lo, 2), round(hi, 2)],
             kind: [round(first, 2), round(last, 2)]}
    if glued:
        entry["chars"] = sum(g.chars for g in glued)
    return entry


def reask(units: list[dict], found: list[tuple[float, float]], glued: list[Glued], *, total: float,
          ask: Callable[[float, float], list[dict]], speech: SpeechSpans,
          voiceprints: Voiceprints, known: Sequence[tuple[float, float]] = ()) -> tuple[list[dict], list[dict]]:
    """`units` with every collapse's and glued word's units replaced by a re-ask of its span, and one log entry per
    span. `found` is `collapses`' (first start, last start) runs, `glued` `unglued`'s words; spots closer than
    `repair.MERGE_WITHIN` are one span. `ask(lo, hi)` is Scribe's units for [lo, hi], absolute times. The span's
    words go; the re-ask's words between the kept words either side come in, less an edge that repeats them
    (`repair.trim_echo`). A failed glued span keeps its split units (`via: split`). Each span is padded out to a
    Silero pause (`cuts.padded`); `known` holds the spans already asked (cached), taken where they fall in its ranges."""
    spans = _spans(found, glued)
    # a glued word's one label is not a voice: its units do not teach the voices
    suspect = [(g.first, g.last) for g in glued]
    voices = Voices(speech.pcm, voiceprints, R.blocks([u for u in units
                                                       if not any(a <= u["start"] <= b for a, b in suspect)]))
    log, failed = [], []
    pauses = cuts.pauses_of(speech.spans()) if spans else []
    breaks = cuts.word_breaks(units) if spans else []
    for k, (first, last, gs) in enumerate(spans, 1):
        lo, hi = cuts.padded(first, last, REASK_PAD, pauses, total, known, breaks=breaks)
        entry = _entry(first, last, gs, lo, hi)
        log.append(entry)
        try:
            words = ask(lo, hi)
        except CacheMiss:
            entry["failed"] = "not cached (calls off)"
        except Exception as error:  # noqa: BLE001 -- a failed spot is given up, the task goes on (owner 2026-11-02)
            entry["failed"] = f"{type(error).__name__}: {error}"[:300]
        else:
            units = _splice(units, words, entry, first, last, lo, hi, voices, apart=f"c{k}" if gs else None)
        if "failed" in entry:
            if gs:
                entry["via"] = "split"
            elif entry["failed"] != "not cached (calls off)":
                failed.append(entry)
    if failed:
        current_reporter().warning("cloud-repair-failed",
                                   f"Scribe 重转 {len(failed)} 处时间戳坍缩失败（{failed[0]['failed']}）",
                                   impact="这几处的台词照原样留着（时间挪到了别处）；metadata 里记了是哪几处")
    return units, log


def _splice(units: list[dict], words: list[dict], entry: dict, first: float, last: float, lo: float, hi: float,
            voices: Voices, *, apart: str | None) -> list[dict]:
    """`units` with those starting in [first, last] replaced by the re-ask's `words` in the hole, `entry` updated;
    unchanged, with `entry["failed"]`, when nothing comes back for the hole. The re-ask's speakers are matched to the
    transcript's by voice (`repair.relabel`); with `apart`, one matched to no one with `relink.RESCUE_MIN` s of voice
    is a new person `<apart>:<label>` (the judge reads `c<k>:` as a re-asked piece's), else unknown; with no
    voiceprints to tell, every label with a block (`repair.blocks`) is a new person."""
    kept = [u for u in units if not first <= u["start"] <= last]
    edge = max((u for u in kept if u["start"] < first), key=lambda u: u["end"], default=None)
    a = edge["end"] if edge else lo
    b = min((u["start"] for u in kept if u["start"] > last), default=hi)
    # the hole only: the re-ask's words either side of it are the kept words again, a few ms off; a kept
    # one-character word ends within EDGE_SLACK of its start (hina-73's 60 ms 「は」), so its copy is told by start
    floor = max(a - EDGE_SLACK, edge["start"] + EDGE_SLACK) if edge else lo
    inside = [w for w in words if floor <= w["start"] < b - EDGE_SLACK]
    left, right = R.neighbours(kept, a, b)
    start, end = R.trim_echo([w["text"] for w in inside], left, right)
    got = inside[start:end]
    if not got:  # the hole's words are replaced, not added to: keep them rather than lose the lines
        entry["failed"] = "the re-ask heard nothing in the hole"
        return units
    mapping = R.relabel(words, voices)
    if apart is not None:
        blocks = R.blocks(words)
        voiced = voices.voiced(blocks)
        voiced = {block["speaker"] for block in blocks} if voiced is None else voiced
        mapping.update({s: f"{apart}:{s}" for s in sorted(voiced - set(mapping))})
    for w in got:
        w["speaker"] = mapping.get(w["speaker"])
    entry.update(dropped=len(units) - len(kept), inserted=len(got), speakers=mapping)
    at = R.insert_at(kept, a)
    return [*kept[:at], *got, *kept[at:]]
