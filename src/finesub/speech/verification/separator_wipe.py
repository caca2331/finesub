"""Did the vocal separator wipe out most of the voice? Then say so, once.

Whispered speech (ASMR, whispered reading) has no pitch; the separator, trained
on voiced singing and talking, removes it as noise -- 20-60 dB in a 30 s
window, where ordinary speech loses 0-1 dB (docs/bench-baselines.md 17.13,
17.19). Whisper then has nothing to transcribe, and the run ends with an
almost empty subtitle and no error. The joint backends read the unseparated
mix and do not have this failure, so the remedy is a note naming them, not a
repair here: repairing it inside the Whisper chain needs a second decoding
path and still ends below `--asr moss-local` (17.19).

The rule is deliberately strict -- a false note sends a user to rerun a task
for nothing, a missed one leaves them where they were without it:

1. Prescreen, free: at least `MIN_SHARE` of the 30 s windows (and at least
   `MIN_WINDOWS` of them) hold under `LOW_COVERAGE` VAD speech. Ordinary
   content fails here (1 of 187 local runs passes), so it never pays for 2.
2. Level, one decode of the source: a window is *wiped* when it is also
   quiet in the mix (`MIX_FLOOR_DB` < level < `MIX_CEILING_DB`) and the
   separator took at least `MIN_ATTEN_DB` off it. The ceiling is what keeps
   music out: a separator correctly emptying a song or an instrumental bed
   removes 50-80 dB from a mix at -10..-25 dBFS; whispered mixes sit below
   -28. The run passes when wiped windows again reach `MIN_SHARE` and
   `MIN_WINDOWS`.
3. Words, on the windows already found: the referee (Qwen3-ASR) reads up to
   `PROBES` wiped windows of the unseparated mix and must hear words in
   `MIN_HEARD` of them. Quiet ambience (rain, a fireplace) passes 1 and 2;
   this is what tells it from a voice (17.17, 17.18).

Both levels are read as 16 kHz mono through ffmpeg, the way the thresholds
were measured.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STEP_SEC = 30.0
LOW_COVERAGE = 0.05
MIN_ATTEN_DB = 15.0
MIX_FLOOR_DB = -60.0
MIX_CEILING_DB = -28.0
MIN_SHARE = 0.30
MIN_WINDOWS = 4
PROBES = 3
MIN_HEARD = 2
MIN_HEARD_WORDS = 3
_SR = 16000
_WORD = re.compile(r"\w+")


def window_coverage(
    segments: Sequence[Mapping[str, Any]], duration: float, *, step: float = STEP_SEC
) -> list[float]:
    """Share of each whole `step` window covered by VAD segments (a trailing partial window is dropped)."""

    count = int(duration // step)
    covered = [0.0] * count
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        for index in range(max(0, int(start // step)), min(count, int(math.ceil(end / step)))):
            lo, hi = index * step, (index + 1) * step
            covered[index] += max(0.0, min(end, hi) - max(start, lo))
    return [value / step for value in covered]


def _enough(flags: Sequence[bool]) -> bool:
    hits = sum(1 for flag in flags if flag)
    return bool(flags) and hits >= MIN_WINDOWS and hits >= MIN_SHARE * len(flags)


def prescreen(coverage: Sequence[float]) -> bool:
    """The necessary condition that costs nothing: enough near-silent windows on the vocal track."""

    return _enough([value < LOW_COVERAGE for value in coverage])


def window_db(samples: np.ndarray, count: int, *, step: float = STEP_SEC) -> list[float]:
    """Mean power in dBFS of each of the first `count` `step` windows of 16 kHz `samples`."""

    width = int(step * _SR)
    levels = []
    for index in range(count):
        chunk = samples[index * width : (index + 1) * width]
        power = float(np.mean(np.square(chunk, dtype=np.float64))) if chunk.size else 0.0
        levels.append(10.0 * math.log10(power + 1e-30))
    return levels


def wiped_windows(
    mix_db: Sequence[float], vocal_db: Sequence[float], coverage: Sequence[float]
) -> list[int]:
    """Windows that were quiet in the mix, lost `MIN_ATTEN_DB` to the separator, and kept almost no VAD speech."""

    return [
        index
        for index, (mix, vocal, cov) in enumerate(zip(mix_db, vocal_db, coverage))
        if MIX_FLOOR_DB < mix < MIX_CEILING_DB and mix - vocal >= MIN_ATTEN_DB and cov < LOW_COVERAGE
    ]


def probe_windows(wiped: Sequence[int]) -> list[int]:
    """Up to `PROBES` of the wiped windows, spread over them."""

    if len(wiped) <= PROBES:
        return list(wiped)
    return [wiped[int((i + 0.5) * len(wiped) / PROBES)] for i in range(PROBES)]


def heard_words(text: str) -> bool:
    return len(_WORD.findall(str(text or ""))) >= MIN_HEARD_WORDS


def _decode(path: str | Path) -> np.ndarray:
    from ...media.ffmpeg import decode_pcm16k

    return np.frombuffer(decode_pcm16k(path, sample_format="f32le"), dtype=np.float32)


def check(
    source: str | Path,
    vocal: str | Path,
    *,
    segments: Sequence[Mapping[str, Any]],
    duration: float,
    referee: Any,
) -> dict[str, Any]:
    """Run the three steps; the record says how far it got and whether the note is due (`wiped`)."""

    coverage = window_coverage(segments, duration)
    record: dict[str, Any] = {"windows": len(coverage), "wiped": False}
    if not prescreen(coverage):
        record["stopped_at"] = "prescreen"
        return record
    mix = _decode(source)
    mix_db = window_db(mix, len(coverage))
    vocal_db = window_db(_decode(vocal), len(coverage))
    wiped = wiped_windows(mix_db, vocal_db, coverage)
    record["wiped_windows"] = len(wiped)
    if not _enough([True] * len(wiped) + [False] * (len(coverage) - len(wiped))):
        record["stopped_at"] = "level"
        return record
    picks = probe_windows(wiped)
    width = int(STEP_SEC * _SR)
    replies = referee.transcribe_batch([mix[i * width : (i + 1) * width] for i in picks])
    heard = sum(1 for text, _language in replies if heard_words(text))
    record.update(probed=len(picks), heard=heard)
    if heard < min(MIN_HEARD, len(picks)):
        record["stopped_at"] = "referee"
        return record
    record["wiped"] = True
    return record


NOTE_CODE = "separator-wiped-voice"


def note_message(record: Mapping[str, Any]) -> str:
    minutes = record.get("wiped_windows", 0) * STEP_SEC / 60.0
    return (
        f"人声分离把约 {minutes:.0f} 分钟的轻声人声（耳语、ASMR 一类）当噪声去掉了，"
        "这些地方几乎转不出字。这类素材请换成直接读原始混音的转写重跑："
        "--asr moss-local（本机，RTX 30 系列或更新、8 GB 显存以上），或云端的 --asr gemini-35t / moss-pro。"
        "同一个任务目录续跑会沿用这次的识别结果，所以先删掉任务目录或换一个 --name。"
    )
