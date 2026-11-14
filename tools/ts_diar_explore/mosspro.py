"""MOSS-Transcribe-Diarize **Pro** through the MOSI API (closed weights, stronger than the open 0.9B per the vendor).

`POST https://api.mosi.cn/v1/audio/transcriptions`, model `moss-transcribe-diarize-pro`, multipart upload,
up to 60 min per request; async mode + `GET /v1/audio/tasks/{id}` polling so long files do not hold a socket.

Quota: keys in the main checkout `.env` as `MOSS_PRO={label:key,...}`, 1.25 audio hours per key per day; the day
starts at Beijing midnight, but a key refreshes only once the owner logs in to the console (owner 2026-10-30). Price
2 RMB per audio hour (owner 2026-09-24). One key is used until its day is spent, then the next -- never alternating.
Every call is a ledger row (`model=mosspro`).

    python -m tools.ts_diar_explore.mosspro <audio> <out.json> [--offset S] [--format json|diarized_json] [--keyterms a,b]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from .common import MAIN, WT, ledger, probe_duration, quota_day_rows, quota_day_start, quota_gate, save

sys.path.insert(0, str(WT / "src"))
from finesub_bootstrap import secrets  # noqa: E402

API = "https://api.mosi.cn/v1/audio/transcriptions"
TASKS = "https://api.mosi.cn/v1/audio/tasks"
MODEL = "moss-transcribe-diarize-pro"
#: Each key has 1.25 h a day; the local check stops at 1.2 h to stay clear of it (owner 2026-10-30). Beyond 1.25 h
#: the server bills the account balance and refuses with HTTP 402 when there is none (2026-09-25). Going over is the
#: owner's call per run (`MOSSPRO_KEY`, see `pick_key`).
DAILY_HOURS = 1.2
OVER_HOURS = 0.0
#: The console counts more than the files' durations: on 2026-10-30 (one observation) it showed about 5% / 7% of the
#: day left where the ledger said 9.4% / 12.4% (4 and 3 calls), 5-6% over; a fixed extra per call would fit too. Local hours are scaled by this before the check.
SERVER_RATIO = 1.06
#: 60 min of audio has taken <= 5 min; a task still running after this is reported, not waited on.
POLL_DEADLINE = 1800.0


def keys() -> list[tuple[str, str]]:
    rhs = secrets.read_env_file(MAIN / ".env").get("MOSS_PRO", "")
    return [(label or f"k{i}", key) for i, (label, key) in enumerate(secrets.iter_entries(rhs))]


def hours_today(label: str) -> float:
    """As the server counts them (`SERVER_RATIO`)."""
    return SERVER_RATIO * sum(r.get("audio_seconds", 0) for r in quota_day_rows("mosspro", label)
                              if not (isinstance(r.get("status"), int) and r["status"] >= 400)) / 3600  # rejected requests bill nothing


def pick_key(seconds: float) -> tuple[str, str]:
    """First key (file order) this call still fits on, counting the call itself; a key the server has
    refused (402) is done for the quota day. `MOSSPRO_KEY=<label>` names the key and skips the local hours
    check -- only when the owner has allowed going over for that run (it spends account balance)."""
    named = os.environ.get("MOSSPRO_KEY")
    for label, key in keys():
        if named and label != named:
            continue
        refused = any(r.get("status") == 402 for r in quota_day_rows("mosspro", label))
        if not refused and (named or hours_today(label) + SERVER_RATIO * seconds / 3600 <= DAILY_HOURS + OVER_HOURS):
            return label, key
    if named and named not in {label for label, _ in keys()}:
        raise SystemExit(f"no MOSS_PRO key labelled {named!r}")
    raise SystemExit(f"no MOSS_PRO key has {SERVER_RATIO * seconds / 3600:.2f} h (as the console counts) left today -- report first")


def call(audio: Path, *, fmt: str, keyterms: list[str]) -> tuple[dict, str, float]:
    quota_gate("mosspro")
    seconds = probe_duration(audio)
    label, key = pick_key(seconds)
    headers = {"Authorization": f"Bearer {key}"}
    data = {"model": MODEL, "diarize": "true", "response_format": fmt, "async": "true"}
    if keyterms:
        data["keyterms"] = json.dumps(keyterms, ensure_ascii=False)
    t0, status = time.monotonic(), None
    try:
        with audio.open("rb") as f:
            r = requests.post(API, headers=headers, data=data, files={"file": (audio.name, f)}, timeout=900)
        status = r.status_code
        if r.status_code == 402:
            raise SystemExit(f"key {label} refused (HTTP 402) although the local ledger has room: the quota day began "
                             f"at {quota_day_start('mosspro')}; ask the owner to log in to the MOSI console")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:2000]}")
        body = r.json()
        task = body.get("task_id") or body.get("id")
        if not task or "segments" in body:  # answered synchronously
            status = "ok"
            return body, label, time.monotonic() - t0
        deadline = time.monotonic() + POLL_DEADLINE
        while True:
            if time.monotonic() > deadline:
                status = "poll-timeout"
                raise RuntimeError(f"task {task} unfinished after {POLL_DEADLINE:.0f} s")
            time.sleep(max(2.0, float(body.get("retry_after") or 5)))
            q = requests.get(f"{TASKS}/{task}", headers=headers, timeout=120)
            if q.status_code >= 400:
                status = f"poll-http{q.status_code}"
                raise RuntimeError(f"task {task} poll HTTP {q.status_code}: {q.text[:2000]}")
            body = q.json()
            state = str(body.get("status", "")).upper()
            if state in ("SUCCEEDED", "SUCCESS", "COMPLETED", "DONE", "FINISHED"):
                status = "ok"
                return body, label, time.monotonic() - t0
            if state in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
                status = f"task-{state.lower()}"
                raise RuntimeError(f"task {task} {state}: {json.dumps(body, ensure_ascii=False)[:2000]}")
    finally:
        ledger(model="mosspro", config=f"fmt={fmt},keyterms={len(keyterms)}", audio=str(audio),
               audio_seconds=round(seconds, 2), key=label, status=status, wall=round(time.monotonic() - t0, 2))


def result_of(body: dict) -> dict:
    """The transcription payload, wherever the task wrapper put it."""
    for k in ("result", "output", "data", "response"):
        if isinstance(body.get(k), dict) and ("segments" in body[k] or "text" in body[k]):
            return body[k]
    return body


def parse(res: dict, *, offset: float) -> tuple[list[dict], list[dict] | None]:
    segs, words = [], []
    for s in res.get("segments", []) or []:
        segs.append({"start": float(s["start"]) + offset, "end": float(s["end"]) + offset,
                     "speaker": s.get("speaker"), "text": s.get("text", "")})
        for w in s.get("words", []) or []:
            words.append({"start": float(w["start"]) + offset, "end": float(w["end"]) + offset,
                          "speaker": w.get("speaker", s.get("speaker")), "text": w.get("word", w.get("text", ""))})
    for w in res.get("words", []) or []:
        words.append({"start": float(w["start"]) + offset, "end": float(w["end"]) + offset,
                      "speaker": w.get("speaker"), "text": w.get("word", w.get("text", ""))})
    return segs, (sorted(words, key=lambda w: w["start"]) or None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--offset", type=float, default=0.0)
    # `verbose_json` is rejected for the pro model (unsupported_response_format, 2026-09-24).
    ap.add_argument("--format", default="json", choices=["json", "diarized_json"])
    ap.add_argument("--keyterms", default="")
    a = ap.parse_args()
    terms = [t for t in a.keyterms.split(",") if t]
    body, label, wall = call(a.audio, fmt=a.format, keyterms=terms)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".raw.json").write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    segs, words = parse(result_of(body), offset=a.offset)
    save({"model": "mosspro", "config": {"format": a.format, "keyterms": terms, "key": label}, "audio": str(a.audio),
          "offset": a.offset, "duration": probe_duration(a.audio), "wall": round(wall, 2),
          "segments": segs, "words": words}, a.out)
    print(f"mosspro key={label}: {len(segs)} segments, {len(words or [])} words, {wall:.1f}s wall, "
          f"key hours today {hours_today(label):.2f} -> {a.out}")


if __name__ == "__main__":
    main()
