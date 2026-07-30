"""Shared helpers for the Qwen3-ASR / Qwen3-ForcedAligner exploration."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

TARGET_SR = 16000

# --------------------------------------------------------------------- test bed
# The 11 clips every §4.6 number is computed over. Baseline artifacts live in the *other*
# worktree (the production checkout), which is where the pipeline was actually run.
BASELINE_ROOT = Path(
    os.environ.get("QWEN_EXPLORE_BASELINE", Path(__file__).resolve().parents[3] / "asr-playground" / "out")
)
EXPLORE_OUT = Path("out/qwen-explore")

TUNING_CLIPS = (
    "BV1kYLR6AEXv", "BV1UBjq6fEgb", "BV1ySjz6FEzD", "BV1cqLR6hEp3",
    "BV1nxje63ERi", "BV1cJjE6cEt8", "BV1dwjP6LECU", "BV1ojjc6MEAs",
)
NEW_CLIPS = ("yingtao", "yui", "kaguya60")
ALL_CLIPS = TUNING_CLIPS + NEW_CLIPS

# Whisper baselines: the tuning clips come from a split-explorer sweep whose `gap03` variant is
# the production default (asr_align.DEFAULT_GAP_SEC = 0.3); the three cross-distribution sources
# were each run separately.
_BASELINE_ALIGNED = {
    "yingtao": "yingtao/yingtao-aligned.json",
    "yui": "yui-exp/yui-split-aligned.json",
    "kaguya60": "kaguya60/kaguya60-aligned.json",
}


def baseline_aligned(clip: str) -> Path:
    rel = _BASELINE_ALIGNED.get(clip)
    if rel is None:
        rel = f"split-explorer-8bv-20260718/{clip}/{clip}-aligned-gap03.json"
    return BASELINE_ROOT / rel


def vad_json(clip: str) -> Path:
    return EXPLORE_OUT / f"{clip}-vad.json"


def qwen_raw(clip: str) -> Path:
    return EXPLORE_OUT / f"{clip}-Q-rescued-raw.json"


def load_audio_16k(path: str | Path) -> np.ndarray:
    """Load an arbitrary audio file as 16 kHz mono float32 (same rate Qwen expects)."""
    wav, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return wav.astype(np.float32)


def cut(wav: np.ndarray, start: float, end: float, pad: float = 0.0) -> np.ndarray:
    a = max(0, int(round((start - pad) * TARGET_SR)))
    b = min(len(wav), int(round((end + pad) * TARGET_SR)))
    return wav[a:b]


def load_aligned(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- SRT

@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _ts(token: str) -> float:
    h, m, s, ms = _TS.match(token.strip()).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def read_srt(path: str | Path) -> list[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        head = 0
        if lines[0].strip().isdigit():
            head = 1
        if "-->" not in lines[head]:
            continue
        a, b = lines[head].split("-->")
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=_ts(a),
                end=_ts(b),
                text="\n".join(lines[head + 1 :]).strip(),
            )
        )
    return cues


# ------------------------------------------------------------------- text metrics

_PUNCT_KEEP = re.compile(r"[^\w぀-ヿ一-鿿]", flags=re.UNICODE)


def normalize_ja(text: str) -> str:
    """NFKC + drop punctuation/whitespace; the unit of comparison is the character."""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_KEEP.sub("", text)
    return text


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref_n, hyp_n = normalize_ja(ref), normalize_ja(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return edit_distance(ref_n, hyp_n) / len(ref_n)
