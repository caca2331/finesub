"""35t Live (`gemini-3.5-transcribe-live`): the text of a short span of audio, over a WebSocket.

Not counted against 35t's daily requests, so the 35t repairs ask it first (`docs/joint-asr.md`「退化的修补」).
At most SESSIONS sessions at once (`docs/gemini35-transcribe.md`「Live 的三条必须」): over that the server
closes the connection with 1011 while it is set up, and the attempt is made again. Server-side VAD is off and
the span is one manual activity. The answer ends with `generationComplete`, which comes the moment its text
does, in one message (2026-11-03: a 29 s span with 12 s of silence in it too; until then the server sends a
textless message every 0.2 s); an empty answer, on silence, ends with nothing more for GRACE s. Text with no
`generationComplete` by then, the server closing the session first, or an `error` / `goAway` message is a
failure and is retried, never an answer (a 100 s span: `turnComplete` with no text, then closed). Text only --
no times, no speakers, no language -- and not reproducible: callers cache what it says.
The key goes in a request header, and every error is scrubbed of it.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any, Callable

from finesub import api_keys
from finesub.api_http import websocket_kwargs

HOST = "generativelanguage.googleapis.com"
URL = f"wss://{HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
MODEL = "gemini-3.5-transcribe-live"
SR = 16000
SESSIONS = 5
GRACE = 6.0
ATTEMPTS = 4
CHUNK = 3200  # bytes: 0.1 s of 16 kHz s16le
#: Silence after the span, so its last word is not cut at the activity's end (the probe's setting).
TAIL = 1.0
#: The task's language as Live's BCP-47 code; another language is sent without one (not probed).
LANGUAGE_CODES = {"ja": "ja-JP", "zh": "zh-CN", "en": "en-US"}

#: The process's sessions: batch items share the cap.
_SLOTS = threading.BoundedSemaphore(SESSIONS)


class LiveError(RuntimeError):
    pass


def setup_message(language: str | None) -> dict:
    transcription: dict[str, Any] = {"mode": "VERBATIM"}
    code = LANGUAGE_CODES.get((language or "").lower())
    if code:
        transcription["languageCodes"] = [code]
    return {"setup": {"model": f"models/{MODEL}", "generationConfig": {"responseModalities": ["TEXT"]},
                      "inputAudioTranscription": transcription,
                      "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}}}}


def _connect(url: str, **options: Any):
    import websocket

    return websocket.create_connection(url, **options)


class Live:
    """`live(pcm, language=...)` -> the text Live heard in 16 kHz mono s16le `pcm`. The first key of the
    `GEMINI_FREE` pool: Live takes no daily requests, so there is no key to move on to."""

    def __init__(self, *, entries=None, connect: Callable[..., Any] = _connect,
                 sleep: Callable[[float], None] = time.sleep):
        self._entries = entries
        self._connect = connect
        self._sleep = sleep

    def _key(self) -> str:
        if self._entries is None:
            self._entries = api_keys.resolve_pool(api_keys.GEMINI_FREE_POOL, api_keys.read_dotenv())
        if not self._entries:
            raise LiveError("35t Live needs a GEMINI_FREE key")
        return self._entries[0].key

    def __call__(self, pcm: bytes, *, language: str | None) -> str:
        key = self._key()
        last = ""
        for attempt in range(ATTEMPTS):
            try:
                with _SLOTS:
                    return self._once(pcm + b"\x00" * int(SR * 2 * TAIL), key, language)
            except ImportError:  # no websocket-client: nothing a retry fixes
                raise LiveError("35t Live needs websocket-client (pip install -e \".[cloud]\")") from None
            except Exception as error:  # noqa: BLE001 -- 1011 while connecting is the session cap; any failure is retried
                last = f"{type(error).__name__}: {error}".replace(key, "<key>")
            if attempt + 1 < ATTEMPTS:
                self._sleep(2.0 + 3.0 * attempt)
        raise LiveError(f"35t Live failed {ATTEMPTS} times; last: {last[:300]}")

    def _once(self, pcm: bytes, key: str, language: str | None) -> str:
        ws = self._connect(URL, header=[f"x-goog-api-key: {key}"], timeout=60, **websocket_kwargs(HOST))
        try:
            ws.send(json.dumps(setup_message(language)))
            if "setupComplete" not in str(ws.recv()):
                raise LiveError("no setupComplete")
            ws.send(json.dumps({"realtimeInput": {"activityStart": {}}}))
            for i in range(0, len(pcm), CHUNK):
                ws.send(json.dumps({"realtimeInput": {"audio": {
                    "mimeType": f"audio/pcm;rate={SR}", "data": base64.b64encode(pcm[i:i + CHUNK]).decode("ascii")}}}))
            ws.send(json.dumps({"realtimeInput": {"activityEnd": {}}}))
            ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
            ws.settimeout(GRACE)
            text: list[str] = []
            while True:
                try:
                    message = ws.recv()
                except Exception as error:  # noqa: BLE001 -- only the grace timeout ends an empty answer
                    if type(error).__name__ not in ("WebSocketTimeoutException", "TimeoutError"):
                        raise
                    if text:
                        raise LiveError(f"text but no generationComplete in {GRACE:.0f} s") from None
                    break
                if not message:  # the server closed the session before the answer was over
                    raise LiveError("the session was closed mid-answer")
                data = json.loads(message)
                if "error" in data or "goAway" in data:
                    raise LiveError(f"server ended the session: {json.dumps(data)[:200]}")
                content = data.get("serverContent") or {}
                part = content.get("inputTranscription") or {}
                if part.get("text"):
                    text.append(part["text"])
                if content.get("generationComplete"):
                    break
            return "".join(text)
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 -- closing a dead socket
                pass
