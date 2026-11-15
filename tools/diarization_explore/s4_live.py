"""Plan 5h.8: S4 through gemini-3.5-transcribe (Live), with and without a speaker anchor.

Same chunks / real spans / decoys / pairs / criteria as 5h.4 (s4_llm). Live has no
timestamps, so each call's text belongs to the span it was asked about, and an
anchor's own transcript is stripped from the front of the anchored result.

  L0  the span (+-0.3 s), transcribed twice (Live is not reproducible: the second
      run is L1's text-volume control, audit 2.8)
  L1  per track (the span's two busiest tracks; decoys: its one track + a silent one):
      that track's own single-speaker speech from elsewhere (<=4 s) + 0.6 s silence
      + the span

  --ask       run the calls (API only, no GPU), cached per call
  --score     the 5h.4 criteria, per arm, plus which anchor helped
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np

import s4_headroom as ref
import s4_llm as llm
import s4_projection as proj

URL = ("wss://generativelanguage.googleapis.com/ws/"
       "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=")
MODEL, LANGUAGE = "models/gemini-3.5-transcribe-live", "zh-CN"
SPAN_PAD, ANCHOR_CLEAR, ANCHOR_MIN, ANCHOR_TOTAL, GAP = 0.3, 3.0, 1.0, 4.0, 0.6
SR = 16000
WORKERS = 4  # the Live wall is 5 concurrent sessions (docs/gemini35-transcribe.md)


def api_key():
    from finesub.llm import llm_runtime
    from finesub import api_keys

    entry, _ = api_keys.first_enabled_gemini_entry(llm_runtime._read_dotenv())
    return entry.key


def live(audio: np.ndarray, key: str, pad=1.0, grace=6.0, attempts=4):
    """The exploration's `live_fast`, fed a float array instead of a file."""
    import websocket

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes() + b"\x00" * int(SR * 2 * pad)
    for attempt in range(attempts):
        try:
            ws = websocket.create_connection(URL + key, timeout=60)
            ws.send(json.dumps({"setup": {"model": MODEL, "generationConfig": {"responseModalities": ["TEXT"]},
                                          "inputAudioTranscription": {"languageCodes": [LANGUAGE], "mode": "VERBATIM"},
                                          "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}}}}))
            if "setupComplete" not in str(ws.recv()):
                raise RuntimeError("setup")
            ws.send(json.dumps({"realtimeInput": {"activityStart": {}}}))
            for i in range(0, len(pcm), 3200):
                ws.send(json.dumps({"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=16000",
                                                                 "data": base64.b64encode(pcm[i:i + 3200]).decode()}}}))
            ws.send(json.dumps({"realtimeInput": {"activityEnd": {}}}))
            ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
            ws.settimeout(grace)
            text = []
            while True:
                try:
                    message = ws.recv()
                except Exception:
                    break
                if not message:
                    break
                try:
                    data = json.loads(message)
                except Exception:
                    continue
                content = data.get("serverContent") or {}
                chunk = (content.get("inputTranscription") or {}).get("text")
                if chunk:
                    text.append(chunk)
                if content.get("generationComplete"):
                    break
            try:
                ws.close()
            except Exception:
                pass
            return "".join(text)
        except Exception as error:  # connection-time failures (the concurrency wall) are retried
            if attempt == attempts - 1:
                return None
            time.sleep(2 + 3 * attempt)
    return None


def strip_anchor(combined: str, anchor: str) -> str:
    """Drop the anchor's transcript from the front of the combined one (best prefix match)."""
    a, c = ref.normalize(anchor), ref.normalize(combined)
    if not a:
        return c
    best_k, best = 0, 0.0
    for k in range(0, min(len(c), int(len(a) * 1.5) + 2) + 1):
        score = difflib.SequenceMatcher(None, c[:k], a).ratio()
        if score > best:
            best_k, best = k, score
    return c[best_k:] if best >= 0.5 else c


def single_runs(active, track):
    alone = active[:, track] & (active.sum(axis=1) == 1)
    runs, start = [], None
    for index, flag in enumerate(np.append(alone, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * proj.FRAME >= ANCHOR_MIN:
                runs.append((start * proj.FRAME, index * proj.FRAME))
            start = None
    return runs


def anchor_audio(audio, active, track, start, end):
    runs = [r for r in single_runs(active, track) if r[1] < start - ANCHOR_CLEAR or r[0] > end + ANCHOR_CLEAR]
    runs.sort(key=lambda r: min(abs(r[0] - end), abs(r[1] - start)))
    pieces, total = [], 0.0
    for s, e in runs:
        take = min(e - s, ANCHOR_TOTAL - total)
        if take <= 0:
            break
        pieces.append(audio[int(s * SR): int((s + take) * SR)])
        total += take
    return np.concatenate(pieces) if pieces else None


def jobs_for(session):
    """Every Live call this session needs, as (job_id, kind, span, track)."""
    import soundfile

    audio, rate = soundfile.read(str(proj.ALI / session / f"{session}-vocal.ogg"), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    assert rate == SR
    tracks = proj.timeline(session)
    active = proj.activity(tracks, int(len(audio) / SR / proj.FRAME) + 2)
    jobs = []
    for chunk in llm.chunks(session):
        for span in chunk["spans"]:
            s, e = span["start"], span["end"]
            clip = audio[max(0, int((s - SPAN_PAD) * SR)): int((e + SPAN_PAD) * SR)]
            sid = f"{s:.2f}-{e:.2f}"
            for rep in (0, 1):
                jobs.append((f"{sid}/L0/{rep}", clip))
            first, last = int(s / proj.FRAME), int(np.ceil(e / proj.FRAME))
            busy = sorted(range(active.shape[1]), key=lambda k: -int(active[first:last, k].sum()))
            live_tracks = [k for k in busy if active[first:last, k].any()][:2]
            if span["kind"] == "decoy":  # its one track + one that is silent here
                live_tracks = live_tracks[:1] + [k for k in busy if k not in live_tracks][:1]
            for k in live_tracks:
                anchor = anchor_audio(audio, active, k, s, e)
                if anchor is None:
                    continue
                jobs.append((f"{sid}/A/{k}", anchor))
                jobs.append((f"{sid}/L1/{k}", np.concatenate([anchor, np.zeros(int(GAP * SR), np.float32), clip])))
    return jobs


def ask():
    key = api_key()
    lock = threading.Lock()
    for session in ref.SESSIONS:
        path = proj.ALI / session / "live.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        todo = [(j, a) for j, a in jobs_for(session) if j not in done]
        print(session, "calls to make", len(todo), "cached", len(done), flush=True)

        def one(job):
            job_id, audio = job
            text = live(audio, key)
            with lock:
                done[job_id] = text
                if len(done) % 100 == 0:
                    path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            return text is None

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            failures = sum(pool.map(one, todo))
        path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        print(session, "failed calls", failures, flush=True)


def score(result="live.json"):
    arms = {"L0x1": [], "L0x2": [], "L1": []}
    spurious = {arm: [0, 0] for arm in arms}
    decoy = {arm: [0, 0] for arm in arms}
    anchor_side = {"worse": [], "better": []}
    base_rows = []
    failures = total_calls = 0
    for session in ref.SESSIONS:
        results = json.loads((proj.ALI / session / result).read_text(encoding="utf-8"))
        total_calls += len(results)
        failures += sum(v is None for v in results.values())
        references = ref.reference(session)
        words = ref.hypothesis(session)
        tracks = proj.timeline(session)
        active = proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)
        spans = [s for c in llm.chunks(session) for s in c["spans"]]
        texts = {}
        for span in spans:
            sid = f"{span['start']:.2f}-{span['end']:.2f}"
            l0 = [ref.normalize(results.get(f"{sid}/L0/{r}") or "") for r in (0, 1)]
            per_track = {}
            for job, text in results.items():
                if job.startswith(f"{sid}/L1/"):
                    k = int(job.rsplit("/", 1)[1])
                    per_track[k] = strip_anchor(text or "", results.get(f"{sid}/A/{k}") or "")
            texts[sid] = {"L0x1": l0[0], "L0x2": l0[0] + l0[1], "L1": "".join(per_track.values()), "tracks": per_track}
            near = set()
            for r in references:
                if r["start"] < span["end"] + 1 and r["end"] > span["start"] - 1:
                    near |= set(ref.bigrams(r["text"]))
            for arm in arms:
                grams = ref.bigrams(texts[sid][arm])
                n = sum(grams.values())
                bad = sum(c for g, c in grams.items() if g not in near)
                spurious[arm][0] += bad
                spurious[arm][1] += n
                if span["kind"] == "decoy":
                    decoy[arm][1] += 1
                    decoy[arm][0] += n > 0 and bad / n > 0.5
        regions = json.loads((proj.ALI / session / "projection.json").read_text(encoding="utf-8"))
        rows = ref.score(session)
        for a, b in ref.pairs(rows):
            middle = (max(a["start"], b["start"]) + min(a["end"], b["end"])) / 2
            if not any(r["start"] <= middle <= r["end"] for r in regions):
                continue
            worse, better = sorted((a, b), key=lambda m: m["recall"])
            window = ref.window_text(words, worse["start"], worse["end"])
            touching = [s for s in spans if s["kind"] == "real" and s["start"] < worse["end"] + ref.PAD and s["end"] > worse["start"] - ref.PAD]
            base_rows.append(worse["recall"])
            for arm in arms:
                extra = "".join(texts[f"{s['start']:.2f}-{s['end']:.2f}"][arm] for s in touching)
                arms[arm].append(ref.recall(worse["text"], window + extra))
            # Which anchor helped: the track overlapping the worse speaker's frames most, vs the other one.
            for s in touching:
                sid = f"{s['start']:.2f}-{s['end']:.2f}"
                first, last = int(max(s["start"], worse["start"]) / proj.FRAME), int(min(s["end"], worse["end"]) / proj.FRAME) + 1
                per_track = texts[sid]["tracks"]
                if len(per_track) < 2:
                    continue
                own = max(per_track, key=lambda k: int(active[first:last, k].sum()))
                for k, text in per_track.items():
                    side = "worse" if k == own else "better"
                    anchor_side[side].append(ref.recall(worse["text"], window + text) - worse["recall"])
    base = mean(base_rows)
    print(f"[{result}] pairs {len(base_rows)}, calls {total_calls}, failed {failures}")
    for arm, values in arms.items():
        gain = mean(values) - base
        sp = spurious[arm][0] / max(spurious[arm][1], 1)
        dc = decoy[arm][0] / max(decoy[arm][1], 1)
        verdict = "可行" if gain >= 0.05 and sp <= 0.443 and dc <= 0.10 else "不做"
        print(f"  {arm:<5} gain {100 * gain:+.1f} pp | spurious {sp:.3f} | decoy confab {dc:.1%} -> {verdict}")
    print(f"  anchor on the drowned speaker's own track: mean gain {100 * mean(anchor_side['worse'] or [0]):+.1f} pp "
          f"(n={len(anchor_side['worse'])}); on the other track: {100 * mean(anchor_side['better'] or [0]):+.1f} pp "
          f"(n={len(anchor_side['better'])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        jobs = jobs_for(ref.SESSIONS[1])
        key = api_key()
        for job_id, audio in jobs[:6]:
            print(job_id, round(len(audio) / SR, 1), "s ->", repr(live(audio, key)))
    if arguments.ask:
        ask()
    if arguments.score:
        score()


if __name__ == "__main__":
    main()
