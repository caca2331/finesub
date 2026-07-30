"""Is a clip's VAD dropping speech? The diagnostic that resolved the `yingtao` root cause.

`yingtao` was the one cross-distribution source where the segmenter kept cutting inside
utterances, and the failure was attributed to the aligner misplacing words. It was not. The
evidence this script produces:

    --words   words landing inside VAD non-speech, per clip, for *both* aligners. Two independent
              ASR+aligner stacks agreeing that something is there means there *is* something
              there. yingtao 6.0% / 3.0%, every other clip < 1%.
    --levels  the level of the vocal track inside VAD non-speech. yingtao's "silence" sits at
              -48.7 dB against -115 / -180 dB elsewhere: vocal separation left a residual floor,
              the energy VAD's SNR margin is relative to that floor, and quiet speech falls under
              it. Needs the separated `*-vocal.flac`, which only some clips still have.
    --timeline  the same two facts minute by minute for one clip, which is what ties them
              together: the minutes with a raised floor are the minutes with misplaced words.

Run `--levels` / `--timeline` in `qwen-asr` (librosa); `--words` runs anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import ALL_CLIPS, BASELINE_ROOT, baseline_aligned, load_audio_16k, qwen_raw, vad_json

SR = 16000
FRAME = 1600  # 100 ms
MIN_WORD_SEC = 0.05
IN_SILENCE_RATIO = 0.9

# The separated tracks that survived; the split-explorer sweep kept only the aligned JSON.
_VOCAL = {
    "yingtao": "yingtao/yingtao-vocal.flac",
    "yui": "yui/yui-vocal.flac",
    "kaguya60": "kaguya60/kaguya60-vocal.flac",
}


def _non_speech(clip: str) -> tuple[list[tuple[float, float]], dict]:
    vad = json.loads(vad_json(clip).read_text(encoding="utf-8"))
    return [(float(a), float(b)) for a, b in vad["non_speech"]], vad


def _overlap(spans, a: float, b: float) -> float:
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans)


def _in_silence(path: Path, spans) -> tuple[int, int]:
    words = [
        w
        for seg in json.loads(path.read_text(encoding="utf-8"))["segments"]
        for w in (seg.get("words") or [])
        if w["end"] - w["start"] > MIN_WORD_SEC
    ]
    bad = sum(
        1
        for w in words
        if _overlap(spans, w["start"], w["end"]) >= IN_SILENCE_RATIO * (w["end"] - w["start"])
    )
    return bad, len(words)


def _frames_db(wav: np.ndarray, spans) -> np.ndarray:
    out: list[float] = []
    for a, b in spans:
        x = wav[int(a * SR) : int(b * SR)]
        n = len(x) // FRAME
        if n:
            rms = np.sqrt((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(1))
            out.extend(20 * np.log10(np.maximum(1e-9, rms)))
    return np.array(out) if out else np.array([-99.0])


def cmd_words() -> None:
    print(f"{'clip':14s} {'qwen 落静音%':>12s} {'whisper 落静音%':>14s}")
    for clip in ALL_CLIPS:
        spans, _ = _non_speech(clip)
        q, qn = _in_silence(qwen_raw(clip), spans)
        w, wn = _in_silence(baseline_aligned(clip), spans)
        print(f"{clip:14s} {100 * q / max(1, qn):12.2f} {100 * w / max(1, wn):14.2f}")
    print("\n两臂同时超过 ~1% = VAD 在漏语音，不是某个对齐器算错。")


def cmd_levels() -> None:
    print(f"{'clip':14s} {'语音中位dB':>10s} {'语音p10':>8s} {'静音中位dB':>10s} {'p10-静音中位':>12s}")
    for clip in ALL_CLIPS:
        rel = _VOCAL.get(clip)
        if rel is None or not (BASELINE_ROOT / rel).exists():
            continue
        wav = load_audio_16k(BASELINE_ROOT / rel)
        spans, vad = _non_speech(clip)
        speech = _frames_db(wav, [(float(a), float(b)) for a, b in vad["speech"]])
        sil = _frames_db(wav, spans)
        p10 = np.percentile(speech, 10)
        print(
            f"{clip:14s} {np.median(speech):10.1f} {p10:8.1f} {np.median(sil):10.1f} "
            f"{p10 - np.median(sil):12.1f}"
        )
    print("\n最后一列接近 0 或为负 = 分离残留把底噪抬到了语音电平，能量 VAD 无从区分。")


def cmd_timeline(clip: str) -> None:
    rel = _VOCAL.get(clip)
    if rel is None:
        raise SystemExit(f"{clip} 没有保留分离后的 vocal 轨")
    wav = load_audio_16k(BASELINE_ROOT / rel)
    spans, vad = _non_speech(clip)
    words = [
        w["start"]
        for seg in json.loads(qwen_raw(clip).read_text(encoding="utf-8"))["segments"]
        for w in (seg.get("words") or [])
        if w["end"] - w["start"] > MIN_WORD_SEC
        and _overlap(spans, w["start"], w["end"]) >= IN_SILENCE_RATIO * (w["end"] - w["start"])
    ]
    print(f"{clip}: {len(words)} 个词落在 VAD 静音里\n")
    print(f"{'分钟':>4s} {'静音底噪中位dB':>14s} {'std':>7s} {'落静音词数':>10s}")
    for m in range(int(vad["duration"]) // 60 + 1):
        window = [(max(s, m * 60), min(e, (m + 1) * 60)) for s, e in spans]
        window = [(a, b) for a, b in window if b - a > 0.1]
        if not window:
            continue
        f = _frames_db(wav, window)
        n = sum(1 for t in words if m * 60 <= t < (m + 1) * 60)
        print(f"{m:4d} {np.median(f):14.1f} {np.std(f):7.1f} {n:10d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", action="store_true")
    ap.add_argument("--levels", action="store_true")
    ap.add_argument("--timeline", metavar="CLIP")
    args = ap.parse_args()

    if args.words or not (args.levels or args.timeline):
        cmd_words()
    if args.levels:
        print()
        cmd_levels()
    if args.timeline:
        print()
        cmd_timeline(args.timeline)


if __name__ == "__main__":
    main()
