"""MAI-Transcribe-2 through Azure fast transcription, OpenRouter (`--via openrouter`) or Vercel AI Gateway (`--via vercel`).

Azure: authentication and endpoint come from `~/.agents/azure/transcribe.py` (Azure hard rule 1: only its
scripts touch Azure). That script does not expose `timestamps`, so the request is rebuilt here with the same
endpoint, token and multipart shape, adding only `definition` fields its README allows.

OpenRouter (owner 2026-10-29, after the Azure resource went away): the owner's key in `.env` as
`FINESUB_KEY_OPENROUTER_STT` (`name:key`, here `sp_paid_limited`; kept apart from the LLM layer's
`FINESUB_KEY_OPENROUTER`), the STT endpoint with `verbose_json`; it routes to Azure, so diarization is the Azure option under
`provider.options`. `--clean` and `--phrases` have no OpenRouter equivalent and are refused.

Vercel (owner 2026-11-19: development goes this way first; its $5 free credit): the owner's key in `.env` as
`FINESUB_KEY_VERCEL_STT`, the request production's `speech/joint/mai2.py` sends (the audio as 16 kHz mono 64 kbps
MP3, diarization and word times under `providerOptions.azure`); the answer carries Azure's own under
`providerMetadata.azure`, read as an Azure one. `--clean` and `--phrases` are refused here too. `episode.py` runs
this module itself: `TS_DIAR_MAI2_VIA=vercel` and `TS_DIAR_MAI2_PAID_OK=1` set the two flags for a run the owner
approved (2026-11-24: the 魔裁 clip d).

All write the same normalized output (`segments`, `words`, absolute seconds after `--offset`).

    python -m tools.ts_diar_explore.mai2 <audio> <out.json> [--offset S] [--diarize] [--timestamps word|segment|none]
                                          [--locale ja] [--clean] [--phrases a,b] [--via azure|openrouter|vercel] [--paid-ok]
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from .common import ledger, mai2_hours_used, probe_duration, save

OPENROUTER_API = "https://openrouter.ai/api/v1/audio/transcriptions"
#: production's gateway request (`finesub.speech.joint.mai2`, dev 2026-11-19)
VERCEL_API = "https://ai-gateway.vercel.sh/v4/ai/transcription-model"
VERCEL_HEADERS = {"ai-gateway-protocol-version": "0.0.1", "ai-transcription-model-specification-version": "4",
                  "ai-model-id": "microsoft/mai-transcribe-2"}
#: Azure credit left about 1900 audio hours at ~$0.10/h, expiring about 2026-10-23 (owner 2026-10-12); stop with slack.
#: After the credit runs out the calls are paid: ask the owner again before calling, and pass `--paid-ok` for
#: a call the owner approved (2026-10-29: the 13 segmentation-gold spans, `explore/mai2_cloud_asr/gold.py`).
BUDGET_STOP_HOURS = 1800.0
CREDIT_UNTIL = _dt.date(2026, 10, 23)


def definition(*, diarize: bool, timestamps: str, locale: str | None, clean: bool, phrases: list[str]) -> dict:
    options: dict = {"timestamps": timestamps}
    if clean:
        options["transcribeStyle"] = "clean"
    d: dict = {"enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2", "modelOptions": options}}
    if diarize:
        d["diarization"] = {"enabled": True}
    if locale:
        d["locales"] = [locale]
    if phrases:
        d["phraseList"] = {"phrases": phrases}
    return d


def openrouter_body(audio: Path, *, diarize: bool, timestamps: str, locale: str | None) -> dict:
    body: dict = {"model": "microsoft/mai-transcribe-2",
                  "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                                  "format": audio.suffix.lstrip(".").lower()}}
    if timestamps != "none":
        body["response_format"] = "verbose_json"
        body["timestamp_granularities"] = ["segment", "word"] if timestamps == "word" else ["segment"]
    if locale:
        body["language"] = locale.split("-")[0]
    if diarize:
        body["provider"] = {"options": {"azure": {"diarization": {"enabled": True}}}}
    return body


def vercel_body(audio: Path, *, locale: str | None) -> dict:
    """Production's gateway request: the audio as its upload MP3, diarization and word times for Azure."""
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = Path(tmp) / "window.mp3"
        subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(audio), "-vn", "-ac", "1",
                        "-ar", "16000", "-b:a", "64k", str(mp3)], check=True)
        data = base64.b64encode(mp3.read_bytes()).decode("ascii")
    azure: dict = {"timestamps": "word", "diarization": {"enabled": True}}
    if locale:
        azure["locales"] = [locale]
    return {"audio": data, "mediaType": "audio/mpeg", "providerOptions": {"azure": azure}}


def env_key(name: str) -> str:
    """The first key of `.env`'s `name`, through the project's only `.env` parser (it decrypts)."""
    from finesub.api_keys import parse_key_list, parse_key_map
    from finesub.paths import resolve_env_file
    from finesub_bootstrap import secrets

    env = resolve_env_file()
    raw = secrets.read_env_file(env).get(name, "") if env else ""
    keys = [k for _, k in parse_key_map(raw)] or parse_key_list(raw)
    if not keys:
        raise SystemExit(f"mai2: {name} is not configured")
    return keys[0]


def call(audio: Path, defn: dict, *, config: str, paid_ok: bool = False, via: str = "azure",
         body: dict | None = None) -> dict:
    """`defn` is the Azure definition, `body` the OpenRouter or Vercel request; only the one for `via` is used."""
    if not paid_ok and (via != "azure" or _dt.date.today() > CREDIT_UNTIL):
        raise SystemExit("mai2: this call is paid (Azure credit ran out about "
                         f"{CREDIT_UNTIL}; OpenRouter and Vercel always) -- ask the owner, then pass --paid-ok")
    seconds = probe_duration(audio)
    if mai2_hours_used() + seconds / 3600 > BUDGET_STOP_HOURS:
        raise SystemExit(f"mai2 budget: {mai2_hours_used():.2f} h used, stop at {BUDGET_STOP_HOURS} h -- report first")
    t0 = time.monotonic()
    status = None
    try:
        if via == "openrouter":
            r = requests.post(OPENROUTER_API, headers={"Authorization": f"Bearer {env_key('FINESUB_KEY_OPENROUTER_STT')}"},
                              json=body, timeout=600)
        elif via == "vercel":
            r = requests.post(VERCEL_API, headers={"Authorization": f"Bearer {env_key('FINESUB_KEY_VERCEL_STT')}",
                                                   **VERCEL_HEADERS}, json=body, timeout=1800)
        else:
            sys.path.insert(0, str(Path.home() / ".agents" / "azure"))
            import transcribe as azure

            with audio.open("rb") as f:
                r = requests.post(
                    f"{azure.ENDPOINT}/speechtotext/transcriptions:transcribe?api-version=2025-10-15",
                    headers={"Authorization": f"Bearer {azure._token()}"},
                    files={"audio": f, "definition": (None, json.dumps(defn))},
                    timeout=1800,
                )
        status = r.status_code
        r.raise_for_status()
        if via != "vercel":
            return r.json()
        azure = (r.json().get("providerMetadata") or {}).get("azure")
        phrases = azure.get("phrases") if isinstance(azure, dict) else None
        if not isinstance(phrases, list) or any((p.get("text") or "").strip() and not p.get("words") for p in phrases):
            raise RuntimeError(f"Vercel answered without Azure's phrases and word times: {r.text[:2000]}")
        return azure
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {status}: {e.response.text[:2000]}") from None
    finally:
        ledger(model="mai2", config=f"via={via},{config}", audio=str(audio), audio_seconds=round(seconds, 2),
               status=status, wall=round(time.monotonic() - t0, 2))


def parse_openrouter(resp: dict, *, offset: float) -> tuple[list[dict], list[dict] | None]:
    def spk(x: dict):
        return None if x.get("speaker") is None else f"S{x['speaker']}"

    segments = [{"start": s["start"] + offset, "end": s["end"] + offset, "speaker": spk(s), "text": s.get("text", ""),
                 "locale": resp.get("language")} for s in resp.get("segments") or []]
    words = [{"start": w["start"] + offset, "end": w["end"] + offset, "speaker": spk(w), "text": w.get("word", "")}
             for w in resp.get("words") or []]
    return segments, (words or None)


def parse(resp: dict, *, offset: float) -> tuple[list[dict], list[dict] | None]:
    segments, words = [], []
    for ph in resp.get("phrases", []):
        spk = ph.get("speaker")
        spk = None if spk is None else f"S{spk}"
        start = ph["offsetMilliseconds"] / 1000 + offset
        segments.append({"start": start, "end": start + ph["durationMilliseconds"] / 1000,
                         "speaker": spk, "text": ph.get("text", ""), "locale": ph.get("locale")})
        for w in ph.get("words", []) or []:
            ws = w["offsetMilliseconds"] / 1000 + offset
            words.append({"start": ws, "end": ws + w["durationMilliseconds"] / 1000, "speaker": spk, "text": w["text"]})
    return segments, (words or None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--timestamps", default="word", choices=["word", "segment", "none"])
    ap.add_argument("--locale")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--phrases", default="")
    ap.add_argument("--via", default=os.environ.get("TS_DIAR_MAI2_VIA", "azure"),
                    choices=["azure", "openrouter", "vercel"])
    ap.add_argument("--paid-ok", action="store_true", default=os.environ.get("TS_DIAR_MAI2_PAID_OK") == "1",
                    help="the owner approved paying for this call")
    a = ap.parse_args()
    phrases = [p for p in a.phrases.split(",") if p]
    if a.via != "azure" and (a.clean or phrases):
        ap.error(f"--clean and --phrases have no {a.via} equivalent")
    if a.via == "vercel" and (not a.diarize or a.timestamps != "word"):
        ap.error("the Vercel request is production's: diarized, word times")
    defn = definition(diarize=a.diarize, timestamps=a.timestamps, locale=a.locale, clean=a.clean, phrases=phrases)
    body = (openrouter_body(a.audio, diarize=a.diarize, timestamps=a.timestamps, locale=a.locale) if a.via == "openrouter"
            else vercel_body(a.audio, locale=a.locale) if a.via == "vercel" else None)
    config = f"ts={a.timestamps},diar={int(a.diarize)},locale={a.locale},clean={int(a.clean)},phrases={len(phrases)}"
    t0 = time.monotonic()
    resp = call(a.audio, defn, config=config, paid_ok=a.paid_ok, via=a.via, body=body)
    wall = time.monotonic() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".raw.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
    segments, words = (parse_openrouter if a.via == "openrouter" else parse)(resp, offset=a.offset)
    sent = defn if a.via == "azure" else {k: v for k, v in body.items() if k not in ("input_audio", "audio")}
    save({"model": "mai2", "via": a.via, "config": sent, "audio": str(a.audio), "offset": a.offset,
          "duration": probe_duration(a.audio), "wall": round(wall, 2), "segments": segments, "words": words}, a.out)
    print(f"mai2 {config}: {len(segments)} segments, {len(words or [])} words, {wall:.1f}s wall, "
          f"budget used {mai2_hours_used():.2f} h -> {a.out}")


if __name__ == "__main__":
    main()
