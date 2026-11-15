"""Does the Gemini API cut a single-frame audio mp4 the way agy does, and what is the cheapest way to have it all heard?
(2026-10-25, free Gemini 3.5 Flash Lite, the files `agy_media_cut_probe.py` built)

  single7 / single20   one single-frame mp4 of 7 / 20 minutes (agy hears 5 / 15)
  framed20q            20 minutes with a black frame every 300 s (agy hears all of it)
  four5                four single-frame 5-minute mp4s
  audio20              the same 20 minutes as plain audio (m4a, `audio/mp4`): no video track at all
  framed20h / framed20m / framed20f   a black frame every 150 s / 60 s / 4 s
  ends20               two black frames, at the start and at 1199 s
  short20              one black frame lasting 1 s, then the audio goes on alone for 20 minutes
  acc11s / acc11e / acc11q / acc11f / acc11v / acc11p   the acceptance check of `agy_media_cut_probe.py` (11 minutes, `A.score`)
  single20lo / ends20lo / framed20flo   single20 / ends20 / framed20f with the API's `videoMetadata.fps` at 0.01
                       (API only: agy cannot set it)

Each arm: countTokens (free) with the per-modality breakdown, then one generateContent asking for the same quotes as
the agy probe, with its usage. Answers are checked by hand against the window's ASR.

    python tools/diarization_explore/api_media_cut_probe.py count|ask [arm,...]   # appends to <MAIN>/tmp/agy-media-cut/api-probes.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agy_media_cut_probe as A  # noqa: E402

from finesub.llm import llm_runtime  # noqa: E402
from finesub.llm.media_upload import upload_gemini_file  # noqa: E402
from finesub import api_keys  # noqa: E402

MODEL = "gemini-3.5-flash-lite"
API = "https://generativelanguage.googleapis.com/v1beta"
OUT = A.WORK / "api-probes.json"
#: seconds between generate calls: the free tier's per-minute token limit is 250k
PAUSE = 40
FILES = {
    "single7": ["full7.mp4"],
    "single20": ["full20.mp4"],
    "framed20q": ["full20q.mp4"],
    "four5": [f"q{k}.mp4" for k in range(4)],
    "audio20": ["full20.m4a"],
    "framed20h": ["full20h.mp4"],
    "framed20m": ["full20m.mp4"],
    "framed20f": ["full20f.mp4"],
    "ends20": ["full20e.mp4"],
    "short20": ["full20s.mp4"],
    **{arm: [f"{arm}.mp4"] for arm in ("acc11s", "acc11e", "acc11q", "acc11f", "acc11v")},
    "acc11p": [f"acc11p-{k}.mp4" for k in range(3)],
    "acc11pad": ["acc11pad.mp4"],
    "acc11end": ["acc11end.mp4"],
    **{arm: [f"{arm}.mp4"] for arm in A.TINY_TRACKS},
    "single20lo": ["full20.mp4"],
    "ends20lo": ["full20e.mp4"],
    "framed20flo": ["full20f.mp4"],
}
#: arm -> the `videoMetadata.fps` its video parts carry
LOW_FPS = {"single20lo": 0.01, "ends20lo": 0.01, "framed20flo": 0.01}
ASK = {"single7": A.ASK["one7"], "single20": A.ASK["one20"], "framed20q": A.ASK["one20q"], "four5": A.ASK["four5"],
       "audio20": "附件是一段 20 分钟的日语录音。" + A.TWENTY,
       **{arm: A.ASK[arm] for arm in ("acc11s", "acc11e", "acc11q", "acc11f", "acc11v", "acc11p", "acc11pad", "acc11end", *A.TINY_TRACKS)},
       **{arm: A.ASK["one20q"] for arm in ("framed20h", "framed20m", "framed20f", "ends20", "short20", "single20lo", "ends20lo", "framed20flo")}}


def m4a():
    """The 20-minute audio track alone, copied out of the single-frame mp4."""
    target = A.WORK / "full20.m4a"
    if not target.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(A.WORK / "full20.mp4"), "-vn", "-c:a", "copy", str(target)],
                       check=True)


def variants():
    """The video variants the agy probe did not build (rate is frames per second of the black track)."""
    wav = A.WORK / "full20.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(A.WORK / "full20.m4a"), "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
    A.framed("full20h", 1200, "1/150")
    A.ends()
    for arm in ("acc11s", "acc11e", "acc11q", "acc11f", "acc11v", "acc11p", "acc11pad", "acc11end", *A.TINY_TRACKS):
        A.FILES[arm]()
    target = A.WORK / "full20s.mp4"
    if not target.exists():
        # no -t: the output lasts as long as its longest stream, the audio
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=1",
                        "-i", str(A.WORK / "full20.m4a"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
                        str(target)], check=True)


def main():
    step, arms = sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else list(FILES)
    m4a()
    variants()
    entry, tier = api_keys.first_enabled_gemini_entry(llm_runtime._read_dotenv())
    assert tier == "GEMINI_FREE", tier
    head = {"x-goog-api-key": entry.key, "Content-Type": "application/json"}
    with httpx.Client(timeout=600) as client:
        for arm in arms:
            refs = [upload_gemini_file(A.WORK / name, api_key=entry.key) for name in FILES[arm]]
            parts = [{"fileData": {"fileUri": r.file_id, "mimeType": r.mime_type},
                      **({"videoMetadata": {"fps": LOW_FPS[arm]}} if arm in LOW_FPS else {})} for r in refs]
            text = {"text": ASK[arm] + A.TAIL}
            body = {"systemInstruction": {"parts": [{"text": A.SYSTEM}]}, "contents": [{"role": "user", "parts": [*parts, text]}]}
            started = time.time()
            if step == "count":
                reply = client.post(f"{API}/models/{MODEL}:countTokens", headers=head,
                                    json={"contents": body["contents"]}).raise_for_status().json()
                row = {"arm": arm, "step": step, "count": reply}
            else:
                reply = client.post(f"{API}/models/{MODEL}:generateContent", headers=head, json=body).raise_for_status().json()
                cand = (reply.get("candidates") or [{}])[0]
                answer = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
                row = {"arm": arm, "step": step, "answer": answer, "usage": reply.get("usageMetadata"),
                       "model_version": reply.get("modelVersion"), "seconds": round(time.time() - started, 1)}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
            OUT.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")
            if step == "ask":
                time.sleep(PAUSE)


if __name__ == "__main__":
    main()
