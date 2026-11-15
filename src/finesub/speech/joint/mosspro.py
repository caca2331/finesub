"""MOSS-Transcribe-Diarize Pro through the MOSI API.

Segment-level times only: each segment is one speaker's turn, segments of
different speakers overlap and each carries its own words. Up to PART_SECONDS
per call; a longer source is cut into parts overlapping by PART_OVERLAP and
stitched by text (`stitch.stitch(segments=True)`). Within a part:

1. an answer that hit the output cap (about 16 384 tokens, reported as
   `SUCCESS` with a broken last segment) is continued from its last good
   segment, until the part's end;
2. any other gap of GAP s or more is re-requested once, only when Silero hears
   GATE_VOICED s of speech in it (owner 2026-10-30: repair detectable defects,
   spend no quota on real silence; Silero missing quiet speech only skips a
   re-request);
3. an answer in the wrong script (an English source transcribed as Japanese,
   seen once) is asked again once; still wrong, the run stops with an error.

A continuation or a gap re-request numbers its speakers anew; `relink` maps
them onto the people already heard by voice (owner 2026-10-30; `docs/joint-asr.md`).

Keys: the `MOSS_PRO` pool. Each key has 1.25 h a day from Beijing midnight
(refreshed only once the owner logs in to the console); the local account stops
at DAILY_HOURS and counts the way the console does (SERVER_RATIO). With
`[asr] moss_pro_paid` (default on) the pool's last key has no such stop: once
the keys before it are spent it goes on, billed to its prepaid balance, until
the server refuses it (402).
"""

from __future__ import annotations

import json
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Callable

from finesub import api_keys, settings
from finesub.api_http import CONNECT_TIMEOUT_SECONDS, api_http_client
from finesub.media.ffmpeg import extract_upload_mp3, probe_media_duration
from finesub.reporting import current_reporter

from . import MOSS_PRO
from .cache import CallCache, audio_fingerprint
from .model import Transcript, Unit, segment_turns
from .quota import BEIJING, KeyLedger, QuotaExhausted
from .relink import VOICEPRINTS, Relinker, Voiceprints
from .stitch import stitch, windows

MODEL = "moss-transcribe-diarize-pro"
API = "https://api.mosi.cn/v1/audio/transcriptions"
TASKS = "https://api.mosi.cn/v1/audio/tasks"

PART_SECONDS = 3600.0
PART_OVERLAP = 60.0
GAP = 30.0
#: The output cap is 16 384 tokens: two answers stopped at 16 385 / 16 401 with status SUCCESS (2026-10-30).
CAP_TOKENS = 16000
#: Every gap a re-request recovered nothing from had 0 s of Silero speech; the two truncated tails 86 s and 290 s.
GATE_VOICED = 3.0
DAILY_HOURS = 1.2
#: The console counted 5-6% more than the files' durations (2026-10-30).
SERVER_RATIO = 1.06
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


class Caller:
    """One request on the first key whose day still has room for it; a key the server refuses (402) is
    spent for the day and the next one is tried. With `paid` the last key always has room: past its free
    hours the server bills its balance, and a 402 there means the balance is empty -- the run stops without
    marking the key spent, so a rerun after topping up uses it the same day.

    The call goes on the key's account when the key is picked (`KeyLedger.reserve`), so batch items
    transcribing at once never squeeze into the same remaining hour. It is taken back only when the server
    certainly did not take the task: a refusal (402 and every other 4xx), or no connection at all. Anything else -- an answer lost after the upload, a failed poll -- stays counted, since the server
    may have billed it: over-counting only moves to the next key early, under-counting can spend the
    account's balance.
    """

    def __init__(self, *, ledger: KeyLedger | None = None, entries=None, paid: bool | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.ledger = ledger or KeyLedger(MOSS_PRO, MODEL, BEIJING)
        self._entries = entries
        self.paid = settings.ASR_MOSS_PRO_PAID.resolve(paid)
        self._sleep = sleep
        self.refused: list[str] = []

    def entries(self):
        if self._entries is None:
            self._entries = api_keys.resolve_pool(api_keys.MOSS_PRO_POOL, api_keys.read_dotenv())
        return self._entries

    def __call__(self, audio: Path, *, seconds: float) -> dict:
        import httpx

        entries = self.entries()
        billed = entries[-1] if self.paid and entries else None

        def fits(entry, usage: dict) -> bool:
            return entry is billed or SERVER_RATIO * (usage["seconds"] + seconds) / 3600 <= DAILY_HOURS

        while True:
            try:
                entry, day = self.ledger.reserve(entries, fits=fits, seconds=seconds, what="moss-pro (MOSS_PRO)")
            except QuotaExhausted as error:
                if self.refused:
                    raise QuotaExhausted(
                        f"{error}; key(s) {', '.join(self.refused)} were refused (HTTP 402) although the local "
                        "account had room: the day has not refreshed (log in to the MOSI console) or the balance is empty"
                    ) from None
                raise
            try:
                body = self._once(audio, entry.key)
            except (MossProRefused, httpx.ConnectError, httpx.ConnectTimeout):
                self.ledger.refund(entry, seconds, day=day)
                raise
            if body is None:
                if entry is billed:
                    # Not marked spent, on purpose: a rerun after a top-up uses it the same day. The price is that
                    # every later batch item uploads once more before it is refused.
                    self.ledger.refund(entry, seconds, day=day)
                    earlier = f"; key(s) {', '.join(self.refused)} were refused (HTTP 402) before it" if self.refused else ""
                    raise QuotaExhausted(
                        f"moss-pro (MOSS_PRO): key {entry.label}, the paid last key of the pool, was refused (HTTP 402): "
                        "its balance is empty or today's free hours have not refreshed; log in to the MOSI console "
                        "(that refreshes the day), top up if the balance is 0, and rerun "
                        f"(or set [asr] moss_pro_paid = false to stop at the free hours){earlier}"
                    )
                self.refused.append(entry.label)
                self.ledger.refund(entry, seconds, day=day, exhausted=True)
                continue
            return body

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
    """The source's 16 kHz PCM and its Silero VAD speech spans, each computed on first use (only a gap or
    a continuation needs them). The stage makes one per source for MOSS Pro, so its line timing reads the
    same samples; they are kept for the whole stage, about 115 MB per hour of source."""

    def __init__(self, source: Path, spans: list[tuple[float, float]] | None = None, pcm=None):
        self._source = source
        self._spans = spans
        self._pcm = pcm

    def pcm(self):
        """int16 samples."""
        if self._pcm is None:
            import numpy as np

            from finesub.media.ffmpeg import decode_pcm16k

            self._pcm = np.frombuffer(decode_pcm16k(self._source), np.int16)
        return self._pcm

    def voiced(self, a: float, b: float) -> float:
        if self._spans is None:
            self._spans = _silero_spans(self.pcm())
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e in self._spans)


def _silero_spans(pcm) -> list[tuple[float, float]]:
    import numpy as np
    import torch

    from finesub.speech.preprocessing.silero_ghost import silero_exclusive

    wav = torch.from_numpy(pcm.astype(np.float32) / 32768)
    with silero_exclusive() as model:
        # ⚠ Only after the model is loaded: importing the `silero_vad` package the first time sets torch to one
        # thread for the whole process; `silero_model()` imports it under its thread-count guard.
        from silero_vad import get_speech_timestamps

        stamps = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in stamps]


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
    `cache_root/moss-pro/`."""

    src = Path(source)
    cache = CallCache(cache_root, MOSS_PRO, identity=identity(src, language), offline=offline)
    call = caller or Caller()
    total = duration if duration is not None else probe_media_duration(src)
    gate = speech or SpeechSpans(src)
    relink = Relinker(gate.pcm, voiceprints or VOICEPRINTS)
    log: list[dict] = []

    asked: set[str] = set()

    def request(name: str, start: float, end: float) -> dict:
        def ask() -> dict:
            current_reporter().debug("moss-pro call", {"start": round(start, 1), "end": round(end, 1)})
            with tempfile.TemporaryDirectory(prefix="finesub-moss-") as tmp:
                clip = extract_upload_mp3(src, start, end, Path(tmp) / "part.mp3")
                return call(clip, seconds=end - start)

        return cache.fetch(name, ask)

    def checked(start: float, end: float, *, fresh: bool = False) -> tuple[dict, list[dict]]:
        """The span's answer and segments. `fresh` asks again under another name when this transcript has
        already asked the same span (a gap left by a stalled continuation), so a gap's re-request is a real one."""
        name = f"s-{start:.1f}-{end:.1f}"
        if fresh and name in asked:
            name += "-again"
        asked.add(name)
        body = request(name, start, end)
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

    def part(lo: float, hi: float) -> list[dict]:
        body, segs = checked(lo, hi)
        span_body = body
        while completion_tokens(span_body) >= CAP_TOKENS:
            a = max((s["end"] for s in segs), default=lo)
            if hi - a < GAP:
                break
            span_body, got = checked(a, hi)
            extra = [s for s in got if s["start"] >= a - 1e-6]
            if max((s["end"] for s in extra), default=a) <= a:  # no progress: the same request would come back
                log.append({"continue_stalled": [round(a, 1), round(hi, 1)]})
                current_reporter().warning(
                    "moss-pro-stalled", f"MOSS Pro 续调 {a:.0f}–{hi:.0f} s 撞了输出上限却没有一段能用",
                    impact="这一段之后的字幕可能缺失（空档还会照常补问一次）")
                break
            log.append({"continue": [round(a, 1), round(hi, 1)], "recovered": len(got),
                        "tokens": completion_tokens(span_body), "speakers": relink(segs, extra)})
            segs = segs + extra
        for a, b in _gaps(segs, lo, hi):
            voiced = gate.voiced(a, b)
            if voiced < GATE_VOICED:
                log.append({"gap": [round(a, 1), round(b, 1)], "voiced": round(voiced, 1), "skipped": True})
                continue
            gap_body, got = checked(a, b, fresh=True)
            extra = [s for s in got if not any(min(s["end"], t["end"]) - max(s["start"], t["start"]) > 0 for t in segs)]
            log.append({"gap": [round(a, 1), round(b, 1)], "voiced": round(voiced, 1), "recovered": len(extra),
                        "tokens": completion_tokens(gap_body), "speakers": relink(segs, extra)})
            segs = segs + extra
        return sorted(segs, key=lambda s: s["start"])

    spans = windows(0.0, total, PART_SECONDS, PART_OVERLAP)
    parts = [(lo, hi, part(lo, hi)) for lo, hi in spans]
    segs = parts[0][2] if len(parts) == 1 else stitch(parts, segments=True)
    if len(parts) > 1:
        log.append({"parts": [[round(lo, 1), round(hi, 1)] for lo, hi in spans]})
    return Transcript(MOSS_PRO, MODEL, total, "segment", segment_turns([Unit.from_dict(s) for s in segs]), log)
