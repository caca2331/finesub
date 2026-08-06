"""Reference data for scoring a VAD, and an honest account of what each one proves.

speech reference (word timestamps)
    Words must be covered by speech. A word sitting inside a non-speech label is
    speech the ASR never gets to see.

    `fixed.srt`   1263 hand-corrected word timestamps for BV1cqLR6hEp3. The only
                  human-checked timeline available.
    `stable.json` ASR output for the other clips. CIRCULAR for the production VAD --
                  those words exist *because* the production VAD kept those regions,
                  so it cannot reveal speech the production VAD already dropped. It
                  is still valid for the opposite direction: any word here that a
                  *candidate* VAD would discard is a real regression.

filled-pause reference
    `disfluency_gold.json` -- 32 blocks a human called pure pause, 25 blocks a human
    called real word onset. A VAD that excludes the first group without touching the
    second is what "better at rejecting filled pauses" actually means.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

Interval = Tuple[float, float]

_TS = re.compile(r"(\d+):(\d\d):(\d\d),(\d+)\s*-->\s*(\d+):(\d\d):(\d\d),(\d+)")


def _sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


@dataclass
class Word:
    start: float
    end: float
    text: str


def load_word_srt(path: Path) -> List[Word]:
    out: List[Word] = []
    block: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip() == "":
            if len(block) >= 3:
                m = _TS.search(block[1])
                if m:
                    out.append(Word(_sec(*m.groups()[:4]), _sec(*m.groups()[4:]),
                                    " ".join(block[2:]).strip()))
            block = []
        else:
            block.append(line)
    return out


def load_words_stable(path: Path) -> List[Word]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    out: List[Word] = []
    for seg in segs:
        for w in seg.get("words") or []:
            s, e = float(w["start"]), float(w["end"])
            if e > s:
                out.append(Word(s, e, str(w.get("word", ""))))
    return out


# Segment tags written by asr-stabilize. Hallucination and filled-pause segments are
# already dropped by profile 0; drift means the timing itself is not trustworthy, so
# such words cannot be used to judge a VAD boundary.
DRIFT_TAG = "时间漂移"
BAD_TAGS = ("高度疑似幻觉", "高度疑似语气填充词")


def load_valid_words(path: Path, *, drop_drift: bool = True,
                     max_word_sec: float = 2.0,
                     repeat_run: int = 3) -> Tuple[List[Word], dict]:
    """Words trustworthy enough to score a VAD against.

    Drops: segments tagged as hallucination / filled pause / drift, runs of the same
    token repeated `repeat_run`+ times (decoder loops), implausibly long words, and
    pure punctuation. Returns the words plus a count of what each rule removed.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    stats = {"segments": len(segs), "dropped_tag": 0, "dropped_drift": 0,
             "dropped_repeat": 0, "dropped_long": 0, "dropped_punct": 0, "kept": 0}
    out: List[Word] = []
    for seg in segs:
        tags = seg.get("tags") or []
        if any(t in tags for t in BAD_TAGS):
            stats["dropped_tag"] += 1
            continue
        if drop_drift and DRIFT_TAG in tags:
            stats["dropped_drift"] += 1
            continue
        words = [w for w in (seg.get("words") or []) if float(w["end"]) > float(w["start"])]
        texts = [str(w.get("word", "")).strip() for w in words]
        blocked = set()
        i = 0
        while i < len(texts):
            j = i
            while j + 1 < len(texts) and texts[j + 1] == texts[i] and texts[i]:
                j += 1
            if j - i + 1 >= repeat_run:
                blocked.update(range(i, j + 1))
            i = j + 1
        for k, w in enumerate(words):
            t = texts[k]
            if k in blocked:
                stats["dropped_repeat"] += 1
                continue
            if float(w["end"]) - float(w["start"]) > max_word_sec:
                stats["dropped_long"] += 1
                continue
            if not t or all(_is_punct(ch) for ch in t):
                stats["dropped_punct"] += 1
                continue
            out.append(Word(float(w["start"]), float(w["end"]), t))
            stats["kept"] += 1
    return out, stats


def _is_punct(ch: str) -> bool:
    import unicodedata

    return unicodedata.category(ch).startswith("P") or ch.isspace()


@dataclass
class PauseRef:
    filled_pause: List[Interval]
    word_onset: List[Interval]


def load_pause_ref(gold_path: Path) -> PauseRef:
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    fp, wo = [], []
    for c in data["disfluency_candidates"]:
        iv = (float(c["start"]), float(c["end"]))
        if c["label"] == "filled_pause":
            fp.append(iv)
        elif c["label"] == "word_onset":
            wo.append(iv)
    return PauseRef(filled_pause=fp, word_onset=wo)


# --------------------------------------------------------------------------
# coverage maths
# --------------------------------------------------------------------------

def covered(intervals: Sequence[Interval], s: float, e: float) -> float:
    """Seconds of [s, e) covered by `intervals` (assumed sorted, disjoint)."""
    tot = 0.0
    for a, b in intervals:
        if b <= s:
            continue
        if a >= e:
            break
        tot += min(b, e) - max(a, s)
    return tot
