"""gemini-3.5-transcribe through `/v1beta/interactions` (unary). Rules: `docs/gemini35-transcribe.md`.

Quota: free keys, RPD 25 each, not refilled the same day. One key is used until it hits the wall, then the
next (never alternating, so a failure stays attributable). A quota hit is recognised in the body, not only
by status, because the SSE form reports it as HTTP 200.

    python -m tools.ts_diar_explore.g35t <audio> <out.json> [--offset S] [--lang ja-JP] [--no-diar]
                                          [--mode verbatim|smart] [--key LABEL]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import MAIN, WT, g35t_calls_today, ledger, probe_duration, quota_day_rows, quota_gate, save, words_to_segments

sys.path.insert(0, str(WT / "src"))
from finesub_bootstrap import secrets  # noqa: E402

UPLOAD = "https://generativelanguage.googleapis.com/upload/v1beta/files"
INTERACTIONS = "https://generativelanguage.googleapis.com/v1beta/interactions"
RPD = 25


class QuotaExhausted(RuntimeError):
    pass


def keys() -> list[tuple[str, str]]:
    rhs = secrets.read_env_file(MAIN / ".env").get("GEMINI_FREE", "")
    return [(label or f"k{i}", key) for i, (label, key) in enumerate(secrets.iter_entries(rhs))]


def pick_key(label: str | None) -> tuple[str, str]:
    """First key (file order) still under RPD today, or the named one."""
    ks = keys()
    if label:
        named = [lk for lk in ks if lk[0] == label]
        if not named:
            raise SystemExit(f"no GEMINI_FREE key labelled {label!r}")
        if g35t_calls_today(label) >= RPD or _exhausted_today(label):
            raise QuotaExhausted(f"key {label} has reached RPD today")
        return named[0]
    for lk in ks:
        if g35t_calls_today(lk[0]) < RPD and not _exhausted_today(lk[0]):
            return lk
    raise QuotaExhausted("every free key has reached RPD today")


def _exhausted_today(label: str) -> bool:
    return any(r.get("status") == "quota" for r in quota_day_rows("g35t", label))


def _post(url: str, data: bytes, headers: dict, timeout: int):
    req = urllib.request.Request(url, method="POST", data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def upload(path: Path, key: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    size = path.stat().st_size
    with _post(UPLOAD, json.dumps({"file": {"display_name": path.name}}).encode(), {
        "x-goog-api-key": key, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json"}, 120) as r:
        session = r.headers.get("X-Goog-Upload-URL")
    with _post(session, path.read_bytes(), {
        "Content-Length": str(size), "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"}, 900) as r:
        info = json.loads(r.read().decode())
    return info["file"]["uri"], mime


def payload(uri: str, mime: str, *, lang: str | None, diarize: bool, mode: str) -> dict:
    m: dict = {"type": mode}
    if mode == "verbatim":
        m["timestamp_granularities"] = ["word"]
        if diarize:
            m["diarization_mode"] = "speaker"
    cfg: dict = {"mode": m}
    if lang:
        cfg["language_codes"] = [lang]
    return {"model": "gemini-3.5-transcribe",
            "input": [{"type": "audio", "uri": uri, "mime_type": mime}],
            "generation_config": {"transcription_config": cfg}}


def call(audio: Path, *, lang: str | None, diarize: bool, mode: str, key_label: str | None) -> tuple[dict, str]:
    quota_gate("g35t")
    label, key = pick_key(key_label)
    seconds = probe_duration(audio)
    t0 = time.monotonic()
    status = None
    try:
        uri, mime = upload(audio, key)
        body = json.dumps(payload(uri, mime, lang=lang, diarize=diarize, mode=mode)).encode()
        for attempt in range(4):
            try:
                with _post(INTERACTIONS, body, {"x-goog-api-key": key, "Content-Type": "application/json"}, 1800) as r:
                    resp = json.loads(r.read().decode())
                status = "ok"
                break
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if "exceeded your current quota" in text or "requests per day" in text:  # the RPD wall (2026-10-30 form)
                    status = "quota"
                    raise QuotaExhausted(f"key {label}: {text[:300]}") from None
                if e.code == 503 and attempt < 3:
                    # "currently experiencing high demand" -- transient overload, not quota.
                    status = "overload-retry"
                    time.sleep(60)
                    continue
                wait = re.search(r"retry in (\d+(?:\.\d+)?)s", text)
                if e.code == 429 and wait and attempt < 3:
                    # Per-minute input-token limit (10k TPM on free tier, 2026-09-24), not the daily wall.
                    status = "tpm-retry"
                    time.sleep(float(wait.group(1)) + 5)
                    continue
                status = f"http{e.code}"
                raise RuntimeError(f"HTTP {e.code}: {text[:2000]}") from None
        if "exceeded your current quota" in json.dumps(resp.get("error", "")):
            status = "quota"
            raise QuotaExhausted(f"key {label}: {resp['error']}")
        return resp, label
    finally:
        ledger(model="g35t", config=f"mode={mode},diar={int(diarize)},lang={lang}", audio=str(audio),
               audio_seconds=round(seconds, 2), key=label, status=status, wall=round(time.monotonic() - t0, 2))


def _sec(v) -> float:
    if v is None:
        return 0.0
    return float(v) if isinstance(v, (int, float)) else float(str(v).rstrip("s"))


def parse(resp: dict, *, offset: float) -> tuple[list[dict], str]:
    words, text = [], []
    for step in resp.get("steps", []):
        for c in step.get("content", []) or []:
            if c.get("type") == "text" and c.get("text"):
                text.append(c["text"])
            for ann in c.get("annotations", []) or []:
                if ann.get("type") != "word_info":
                    continue
                words.append({"start": _sec(ann.get("start_offset")) + offset,
                              "end": _sec(ann.get("end_offset")) + offset,
                              "speaker": ann.get("speaker"), "text": ann.get("text", "")})
    words.sort(key=lambda w: w["start"])
    return words, "\n".join(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--lang", default="ja-JP", help="'' for auto-detect")
    ap.add_argument("--no-diar", action="store_true")
    ap.add_argument("--mode", default="verbatim", choices=["verbatim", "smart"])
    ap.add_argument("--key")
    a = ap.parse_args()
    t0 = time.monotonic()
    resp, label = call(a.audio, lang=a.lang or None, diarize=not a.no_diar, mode=a.mode, key_label=a.key)
    wall = time.monotonic() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".raw.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
    words, text = parse(resp, offset=a.offset)
    if a.mode == "verbatim" and not words:
        # docs/gemini35-transcribe.md: zero words is a parser bug until proven otherwise.
        print(f"WARNING: zero words parsed; inspect {a.out.with_suffix('.raw.json')} before treating as silence")
    save({"model": "g35t", "config": {"mode": a.mode, "diarize": not a.no_diar, "lang": a.lang, "key": label},
          "audio": str(a.audio), "offset": a.offset, "duration": probe_duration(a.audio), "wall": round(wall, 2),
          "text": text, "segments": words_to_segments(words) if words else [], "words": words or None}, a.out)
    print(f"g35t key={label}: {len(words)} words, {wall:.1f}s wall, calls today {g35t_calls_today()} -> {a.out}")


if __name__ == "__main__":
    main()
