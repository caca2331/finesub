"""MOSS-Transcribe-Diarize Pro through the MOSI API.

Segment-level times only: each segment is one speaker's turn, segments of
different speakers overlap and each carries its own words. A source goes in
pieces of up to PART_SECONDS, each planned onto a key by `fill.py` when it is
reached (free hours first, cut as little as possible, on Silero pauses),
overlapping by 60-90 s and stitched by text (`stitch.stitch(segments=True)`).
Within a piece:

1. an answer that hit the output cap (about 16 384 tokens, reported as
   `SUCCESS` with a broken last segment) ends its piece at its last good
   segment; the next piece starts there, without overlap, and is planned onto
   the keys like any other;
2. any other gap of GAP s or more is re-requested once, only when Silero hears
   GATE_VOICED s of speech in it (owner 2026-10-30: repair detectable defects,
   spend no quota on real silence; Silero missing quiet speech only skips a
   re-request);
3. an answer in the wrong script (an English source transcribed as Japanese,
   seen once) is asked again once; still wrong, the run stops with an error.

A gap re-request numbers its speakers anew: its labels are renamed apart (a
piece's are, by `stitch`), and once the transcript is put together `relink.merge_by_voice`
merges, over all of it, the labels that are one person -- those of the
re-requests, of the parts, and the ones a single call split a person into
(owner 2026-11-09; `docs/joint-asr.md`).

Keys: the `MOSS_PRO` pool. Each key has 1.25 h a day from Beijing midnight
(refreshed only once the owner logs in to the console); the local account stops
at DAILY_HOURS and counts the way the console does (SERVER_RATIO). With
`[asr] moss_pro_paid` (default on) the pool's last key has no such stop: what
the free hours cannot hold goes on it, billed to its prepaid balance.
"""

from __future__ import annotations

import json
import tempfile
import time
import unicodedata
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from finesub import api_keys, settings
from finesub.api_http import CONNECT_TIMEOUT_SECONDS, api_http_client
from finesub.media.ffmpeg import extract_upload_mp3, probe_media_duration
from finesub.reporting import current_reporter

from . import MOSS_PRO, fill, moss_timing
from .cache import CacheMiss, CallCache, audio_fingerprint
from .model import Transcript, Unit, segment_turns
from .quota import BEIJING, KeyLedger, QuotaExhausted
from .relink import REUSE_UNMERGED, VOICEPRINTS, Voiceprints, merge_by_voice, renumber
from .stitch import stitch

MODEL = "moss-transcribe-diarize-pro"
API = "https://api.mosi.cn/v1/audio/transcriptions"
TASKS = "https://api.mosi.cn/v1/audio/tasks"

PART_SECONDS = fill.PART_SECONDS
PART_OVERLAP = fill.OVERLAP[0]
#: What the cached answer of a call records as the key it went on (the log attributes each call to its key).
KEY_FIELD = "_finesub_key"
GAP = 30.0
#: The output cap is 16 384 tokens: two answers stopped at 16 385 / 16 401 with status SUCCESS (2026-10-30).
CAP_TOKENS = 16000
#: Every gap a re-request recovered nothing from had 0 s of Silero speech; the two truncated tails 86 s and 290 s.
GATE_VOICED = 3.0
DAILY_HOURS = 1.2
#: The console counts about the files' durations (owner 2026-11-09); the 2% is room for jitter.
SERVER_RATIO = 1.02
POLL_DEADLINE = 1800.0
#: Languages whose transcript is in Latin script, and those whose is not; others are not checked.
LATIN = {"en", "de", "fr", "es", "it", "pt", "nl", "id", "vi"}
NON_LATIN = {"ja", "zh", "ko", "ru", "ar", "th"}


class MossProError(RuntimeError):
    pass


class MossProRefused(MossProError):
    """The server refused the upload with a 4xx: no task, nothing billed."""


# --- reading an answer ------------------------------------------------------------------------------------


def result_of(body: dict) -> dict:
    """The transcription payload, wherever the task wrapper put it."""
    for k in ("result", "output", "data", "response"):
        if isinstance(body.get(k), dict) and ("segments" in body[k] or "text" in body[k]):
            return body[k]
    return body


def segments_of(body: dict, offset: float) -> list[dict]:
    """Its segments with absolute times. A segment cut at the output cap (end 0) and the run of 0.5 s "..."
    segments a re-request over music can return are not speech and are dropped."""
    out = []
    for s in result_of(body).get("segments", []) or []:
        seg = {"start": float(s["start"]) + offset, "end": float(s["end"]) + offset,
               "speaker": s.get("speaker"), "text": s.get("text", "")}
        if seg["end"] > seg["start"] and any(unicodedata.category(ch)[0] in "LN" for ch in seg["text"]):
            out.append(seg)
    return out


def completion_tokens(body: dict) -> int:
    return int((body.get("usage") or {}).get("completion_tokens") or 0)


def wrong_script(segments: list[dict], language: str | None) -> bool:
    """Whether most letters are not in the script the source language is written in."""
    lang = (language or "").lower()
    if lang not in LATIN | NON_LATIN:
        return False
    letters = [ch for s in segments for ch in s["text"] if unicodedata.category(ch)[0] == "L"]
    if not letters:
        return False
    latin = sum("LATIN" in unicodedata.name(ch, "") for ch in letters) / len(letters)
    return latin < 0.5 if lang in LATIN else latin > 0.5


# --- calling ----------------------------------------------------------------------------------------------


def _client():
    import httpx

    return api_http_client(httpx.Timeout(900.0, connect=CONNECT_TIMEOUT_SECONDS))


class Replan(Exception):
    """A key refused a planned piece (402); the piece is planned again without it."""


@dataclass(frozen=True)
class Reservation:
    entry: api_keys.ApiKeyEntry
    day: str
    piece: fill.Piece


class Caller:
    """Calls on the `MOSS_PRO` pool, each on the key the account picks for it at the moment it is made
    (`KeyLedger.reserve_with`, so batch items transcribing at once never squeeze into the same remaining hour):
    a piece of the source by `fill.plan` (`reserve_piece`, then `send`), a short call (a continuation, a gap,
    a retry) whole into the key with the least room that holds it (`__call__`). With `paid` the last key also
    takes what no free hours hold, billed to its balance.

    A call is taken back off the account only when the server certainly did not take it: a refusal (402 and
    every other 4xx), or no connection at all. Anything else -- an answer lost after the upload, a failed poll --
    stays counted, since the server may have billed it: over-counting only moves to another key early,
    under-counting can spend the account's balance. A free key refused with 402 is spent for the day; the paid
    key refused is left out for the rest of this transcript and not marked spent, so a rerun after topping up
    uses it the same day (owner 2026-11-09).
    """

    def __init__(self, *, ledger: KeyLedger | None = None, entries=None, paid: bool | None = None,
                 sleep: Callable[[float], None] = time.sleep, announce: Callable[[], None] | None = None):
        self.ledger = ledger or KeyLedger(MOSS_PRO, MODEL, BEIJING)
        self._entries = entries
        self.paid = settings.ASR_MOSS_PRO_PAID.resolve(paid)
        self._sleep = sleep
        self._announce = announce
        self.refused: list[str] = []
        self.left_out: str | None = None  # the paid key, after a 402

    def entries(self):
        if self._entries is None:
            self._entries = api_keys.resolve_pool(api_keys.MOSS_PRO_POOL, api_keys.read_dotenv())
        return self._entries

    def keys(self, usages: Sequence[dict]) -> list[fill.Key | None]:
        """Per key, its free room today as `fill` sees it; None for a key that takes nothing."""
        entries = self.entries()
        billed = entries[-1] if self.paid and entries else None
        out = []
        for entry, usage in zip(entries, usages):
            if usage["exhausted"] or entry.label == self.left_out:
                out.append(None)
                continue
            room = max(0.0, DAILY_HOURS * 3600 / SERVER_RATIO - usage["seconds"])
            out.append(fill.Key(room, paid=entry is billed))
        return out

    def reserve_piece(self, total: float, starts: Sequence[tuple[float, float]],
                      pauses: Sequence[tuple[float, float]]) -> Reservation:
        """The next piece of the source, planned against every key's account and reserved on its key in one
        locked step."""

        def choose(usages):
            keys = self.keys(usages)
            usable = [i for i, k in enumerate(keys) if k is not None]
            pieces = fill.plan(total, starts, [keys[i] for i in usable], pauses)
            if not pieces:
                return None
            first = pieces[0]
            i = usable[first.key]
            return i, first.end - first.start, fill.Piece(first.start, first.end, i, first.billed)

        entry, day, piece = self._reserve(choose)
        return Reservation(entry, day, piece)

    def __call__(self, audio: Path, *, seconds: float) -> dict:
        def choose(usages):
            keys = self.keys(usages)
            free = [(k.room, i) for i, k in enumerate(keys) if k is not None and seconds <= k.room]
            if free:
                return min(free)[1], seconds, None
            paid = [i for i, k in enumerate(keys) if k is not None and k.paid]
            return (paid[0], seconds, None) if paid else None

        while True:
            entry, day, _ = self._reserve(choose)
            try:
                return self.send(audio, entry, day, seconds)
            except Replan:
                continue

    def _reserve(self, choose):
        try:
            return self.ledger.reserve_with(self.entries(), choose, what="moss-pro (MOSS_PRO)")
        except QuotaExhausted as error:
            said = [str(error)]
            if self.left_out:
                said.append(f"key {self.left_out}, the paid last key of the pool, was refused (HTTP 402): its balance "
                            "is empty or today's free hours have not refreshed; log in to the MOSI console (that "
                            "refreshes the day), top up if the balance is 0, and rerun (or set [asr] moss_pro_paid = "
                            "false to stop at the free hours)")
            if self.refused:
                said.append(f"key(s) {', '.join(self.refused)} were refused (HTTP 402) although the local account "
                            "had room: the day has not refreshed (log in to the MOSI console) or the balance is empty")
            raise QuotaExhausted("; ".join(said)) from None

    def send(self, audio: Path, entry: api_keys.ApiKeyEntry, day: str, seconds: float) -> dict:
        """The call reserved on `entry` for quota `day` made; `Replan` when the key refused it (402)."""
        import httpx

        if self._announce is not None:
            self._announce()
        try:
            body = self._once(audio, entry.key)
        except (MossProRefused, httpx.ConnectError, httpx.ConnectTimeout):
            self.ledger.refund(entry, seconds, day=day)
            raise
        if body is not None:
            return {**body, KEY_FIELD: entry.label}
        entries = self.entries()
        if self.paid and entry is entries[-1]:
            self.left_out = entry.label
            self.ledger.refund(entry, seconds, day=day)
        else:
            self.refused.append(entry.label)
            self.ledger.refund(entry, seconds, day=day, exhausted=True)
        raise Replan(entry.label)

    def _once(self, audio: Path, key: str) -> dict | None:
        import httpx

        headers = {"Authorization": f"Bearer {key}"}
        data = {"model": MODEL, "diarize": "true", "response_format": "json", "async": "true"}
        with _client() as client:
            with audio.open("rb") as f:
                r = client.post(API, headers=headers, data=data, files={"file": (audio.name, f)})
            if r.status_code == 402:
                return None
            if 400 <= r.status_code < 500:
                raise MossProRefused(f"HTTP {r.status_code}: {r.text[:2000]}")
            if r.status_code >= 500:  # a gateway error may hide a task that was created: stays counted
                raise MossProError(f"HTTP {r.status_code}: {r.text[:2000]}")
            body = r.json()
            task = body.get("task_id") or body.get("id")
            if not task or "segments" in body:  # answered synchronously
                return body
            deadline = time.monotonic() + POLL_DEADLINE
            while True:
                if time.monotonic() > deadline:
                    raise MossProError(f"task {task} unfinished after {POLL_DEADLINE:.0f} s")
                self._sleep(max(2.0, float(body.get("retry_after") or 5)))
                try:
                    q = client.get(f"{TASKS}/{task}", headers=headers, timeout=httpx.Timeout(120.0, connect=CONNECT_TIMEOUT_SECONDS))
                except (httpx.ConnectError, httpx.ConnectTimeout) as error:  # the task exists: stays counted
                    raise MossProError(f"task {task} poll failed: {error}") from error
                if q.status_code >= 400:
                    raise MossProError(f"task {task} poll HTTP {q.status_code}: {q.text[:2000]}")
                body = q.json()
                state = str(body.get("status", "")).upper()
                if state in ("SUCCEEDED", "SUCCESS", "COMPLETED", "DONE", "FINISHED"):
                    return body
                if state in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
                    raise MossProError(f"task {task} {state}: {json.dumps(body, ensure_ascii=False)[:2000]}")


# --- the Silero gate --------------------------------------------------------------------------------------


class SpeechSpans:
    """The source's 16 kHz PCM, Silero's probability per frame and its speech spans, each computed on first use
    (MOSS Pro needs the spans only for a gap or a continuation). The stage makes one per source for the MOSS
    backends, so the transcription and the line timing read the same samples and Silero scores them once; they
    are kept for the whole stage, about 115 MB per hour of source."""

    def __init__(self, source: Path, spans: list[tuple[float, float]] | None = None, pcm=None):
        self._source = source
        self._spans = spans
        self._pcm = pcm
        self._scored = None

    def pcm(self):
        """int16 samples."""
        if self._pcm is None:
            import numpy as np

            from finesub.media.ffmpeg import decode_pcm16k

            self._pcm = np.frombuffer(decode_pcm16k(self._source), np.int16)
        return self._pcm

    def probs(self):
        """Silero's probability per whole 32 ms frame (`moss_timing.frame_probabilities`)."""
        return self._probs()[:len(self.pcm()) // moss_timing.HOP]

    def spans(self) -> list[tuple[float, float]]:
        if self._spans is None:
            self._spans = _silero_spans(self.pcm(), self._probs())
        return self._spans

    def voiced(self, a: float, b: float) -> float:
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e in self.spans())

    def _probs(self):
        if self._scored is None:
            self._scored = moss_timing.frame_probabilities(self.pcm(), tail=True)
        return self._scored


def _silero_spans(pcm, probs=None) -> list[tuple[float, float]]:
    """Silero's speech spans by its own `get_speech_timestamps` (thresholds, minimum lengths, padding), run over
    `probs` (`frame_probabilities(pcm, tail=True)`; scored here when not given)."""
    import numpy as np
    import torch

    if probs is None:
        probs = moss_timing.frame_probabilities(pcm, tail=True)
    # ⚠ Only after the model is loaded (scoring loads it): importing the `silero_vad` package the first time sets
    # torch to one thread for the whole process; `silero_model()` imports it under its thread-count guard.
    from silero_vad import get_speech_timestamps

    wav = torch.from_numpy(pcm.astype(np.float32) / 32768)
    stamps = get_speech_timestamps(wav, _Scored(probs), sampling_rate=16000, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in stamps]


class _Scored:
    """Stands in for the model in `get_speech_timestamps`: each call (one per frame, in order) hands back the next
    probability already scored, so its rules run unchanged without the per-frame model calls."""

    def __init__(self, probs):
        self._next = iter(probs).__next__

    def reset_states(self) -> None:
        pass

    def __call__(self, chunk, sampling_rate):
        return self._next()


def _gaps(segs: list[dict], lo: float, hi: float) -> list[tuple[float, float]]:
    out, reach = [], lo
    for s in sorted(segs, key=lambda s: s["start"]):
        if s["start"] - reach >= GAP:
            out.append((reach, s["start"]))
        reach = max(reach, s["end"])
    if hi - reach >= GAP:
        out.append((reach, hi))
    return out


# --- one source -------------------------------------------------------------------------------------------


def call_name(start: float, end: float) -> str:
    return f"s-{start:.1f}-{end:.1f}"


def main_call(names, reach: float, *, first: bool, capped: bool = False, after: float = -1.0) -> tuple[float, float] | None:
    """(start, end) of the cached main call of the piece after `reach` (the previous piece's end): it starts at 0
    for the first piece, within 0.1 s of `reach` after a piece that hit the output cap there, else OVERLAP s before
    `reach` (a range: the start was a Silero pause, which another machine may place a little apart); it ends past
    `reach` (the previous piece's gap re-requests end inside it); not a `-again` / `-retry`. Of those the latest
    end, then the earliest start: a gap re-request inside this piece ends before its main call. It starts after
    `after` (the previous piece's start), so a piece capped a moment in does not find itself again."""
    # After a cap, 0.1 s either way: an old task's continuation was cut at the unrounded point and named at 0.1 s,
    # and its segments are now read from the name's start, so a later cap point can be a rounding step off.
    lo, hi = (0.0, 0.0) if first else (reach - 0.1, reach + 0.1) if capped else (
        reach - fill.OVERLAP[1], reach - fill.OVERLAP[0])
    best = None
    for name in names:
        m = re.fullmatch(r"s-(\d+\.\d)-(\d+\.\d)", name)
        if not m:
            continue
        a, b = float(m[1]), float(m[2])
        if (max(lo - 0.05, after + 0.05) <= a <= hi + 0.05 and b > reach + 0.05
                and (best is None or (b, -a) > (best[1], -best[0]))):
            best = (a, b)
    return best


def identity(source: Path, language: str | None) -> dict:
    return {"provider": MOSS_PRO, "model": MODEL, "part": PART_SECONDS, "part_overlap": PART_OVERLAP,
            "language": language, "audio": audio_fingerprint(source)}


def transcribe(
    source: str | Path,
    *,
    cache_root: str | Path,
    language: str | None,
    offline: bool = False,
    caller: Caller | None = None,
    duration: float | None = None,
    speech: SpeechSpans | None = None,
    voiceprints: Voiceprints | None = None,
) -> Transcript:
    """The source's MOSS Pro transcript (one turn per segment, by start); every call cached under
    `cache_root/moss-pro/`. The source goes piece by piece: each piece's main call is the one already cached for
    it (`main_call`), else planned and reserved now (`Caller.reserve_piece`), so a rerun keeps the first run's
    cuts whatever the keys' room has become."""

    src = Path(source)
    cache = CallCache(cache_root, MOSS_PRO, identity=identity(src, language), offline=offline)
    call = caller or Caller()
    total = duration if duration is not None else probe_media_duration(src)
    gate = speech or SpeechSpans(src)
    renumbered = 0

    def apart(extra: list[dict]) -> dict[str, str]:
        nonlocal renumbered
        renumbered += 1
        return renumber(extra, renumbered)

    log: list[dict] = []
    keys: list[list] = []  # [call name, the key it went on] (None: cached before keys were recorded)

    asked: set[str] = set()

    def request(name: str, start: float, end: float, reserved: Reservation | None = None) -> dict:
        sending = False

        def ask() -> dict:
            nonlocal sending
            current_reporter().debug("moss-pro call", {"start": round(start, 1), "end": round(end, 1)})
            with tempfile.TemporaryDirectory(prefix="finesub-moss-") as tmp:
                clip = extract_upload_mp3(src, start, end, Path(tmp) / "part.mp3")
                if reserved is None:
                    return call(clip, seconds=end - start)
                sending = True
                return call.send(clip, reserved.entry, reserved.day, reserved.piece.end - reserved.piece.start)

        try:
            body = cache.fetch(name, ask)
        finally:
            if reserved is not None and not sending:  # nothing went out: the reservation goes back
                call.ledger.refund(reserved.entry, reserved.piece.end - reserved.piece.start, day=reserved.day)
        keys.append([name, body.get(KEY_FIELD)])
        return body

    def checked(start: float, end: float, *, fresh: bool = False,
                reserved: Reservation | None = None) -> tuple[dict, list[dict]]:
        """The span's answer and segments. `fresh` asks again under another name when this transcript has
        already asked the same span (a gap left by a stalled continuation), so a gap's re-request is a real one."""
        name = call_name(start, end)
        if fresh and name in asked:
            name += "-again"
        asked.add(name)
        body = request(name, start, end, reserved)
        segs = segments_of(body, start)
        if wrong_script(segs, language):
            log.append({"wrong_script": [round(start, 1), round(end, 1)]})
            body = request(name + "-retry", start, end)
            segs = segments_of(body, start)
            if wrong_script(segs, language):
                raise MossProError(
                    f"MOSS Pro answered {start:.0f}-{end:.0f} s twice in a script that is not {language}'s "
                    "(it has transcribed English as a Japanese translation before); use another --asr backend")
        return body, segs

    def part(lo: float, hi: float, reserved: Reservation | None = None) -> tuple[float, list[dict]]:
        """The piece's segments and where it ends: `hi`, or its last good segment when the answer hit the output
        cap there (the next piece starts at that point)."""
        body, segs = checked(lo, hi, reserved=reserved)
        end = hi
        if completion_tokens(body) >= CAP_TOKENS:
            # one value for the next piece's start, its call name and its lookup on resume
            a = round(max((s["end"] for s in segs), default=lo), 1)
            if a > lo and hi - a >= GAP:
                log.append({"capped": [round(lo, 1), round(hi, 1)], "at": a})
                end = a
            elif a <= lo:  # no progress: the same request would come back
                log.append({"continue_stalled": [round(lo, 1), round(hi, 1)]})
                current_reporter().warning(
                    "moss-pro-stalled", f"MOSS Pro {lo:.0f}–{hi:.0f} s 撞了输出上限却没有一段能用",
                    impact="这一段之后的字幕可能缺失（空档还会照常补问一次）")
        for a, b in _gaps(segs, lo, end):
            voiced = gate.voiced(a, b)
            if voiced < GATE_VOICED:
                log.append({"gap": [round(a, 1), round(b, 1)], "voiced": round(voiced, 1), "skipped": True})
                continue
            gap_body, got = checked(a, b, fresh=True)
            extra = [s for s in got if not any(min(s["end"], t["end"]) - max(s["start"], t["start"]) > 0 for t in segs)]
            log.append({"gap": [round(a, 1), round(b, 1)], "voiced": round(voiced, 1), "recovered": len(extra),
                        "tokens": completion_tokens(gap_body), "speakers": apart(extra)})
            segs = segs + extra
        return end, sorted(segs, key=lambda s: s["start"])

    cached = cache.names()
    pauses: list[tuple[float, float]] | None = None

    def silences() -> list[tuple[float, float]]:
        nonlocal pauses
        if pauses is None:
            pauses = fill.pauses_of(gate.spans())
        return pauses

    parts: list[tuple[float, float, list[dict]]] = []
    capped = False  # the last piece hit the output cap: the next starts where it stopped, without overlap
    while not parts or parts[-1][1] < total - 0.05:
        reach = parts[-1][1] if parts else 0.0
        found = main_call(cached, reach, first=not parts, capped=capped, after=parts[-1][0] if parts else -1.0)
        if found is not None:
            end, segs = part(*found)
            parts.append((found[0], end, segs))
            capped = end < found[1]
            continue
        if offline:
            raise CacheMiss(f"{cache.dir}: no cached call covers {reach:.0f} s on and calls are off")
        starts = [(reach, 0.0)] if capped else fill.next_starts(reach, silences()) if parts else [(0.0, 0.0)]
        while True:
            reserved = call.reserve_piece(total, starts, silences())
            lo, hi = round(reserved.piece.start, 1), round(reserved.piece.end, 1)
            try:
                end, segs = part(lo, hi, reserved)
                break
            except Replan:
                continue
        parts.append((lo, end, segs))
        capped = end < hi
    segs = parts[0][2] if len(parts) == 1 else stitch(parts, segments=True)
    if len(parts) > 1:
        log.append({"parts": [[round(lo, 1), round(hi, 1)] for lo, hi, _ in parts]})
    log.append({"keys": keys})
    merges = merge_by_voice(segs, gate.pcm, voiceprints or VOICEPRINTS, "MOSS Pro", unmerged=REUSE_UNMERGED)
    if merges:
        log.append({"merge": [list(m) for m in merges]})
    return Transcript(MOSS_PRO, MODEL, total, "segment", segment_turns([Unit.from_dict(s) for s in segs]), log)
