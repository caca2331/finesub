"""Where this experiment's inputs live.

The run was done from a worktree, so `assets/`, `data/` and `out/` sat in the
main checkout rather than next to the code -- those directories are gitignored
and exist only once. `FINESUB_DATA_ROOT` points at whichever checkout holds
them; it defaults to the current one, which is right when you run from the main
checkout and wrong from a worktree.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("FINESUB_DATA_ROOT", ".")).resolve()

#: Where the arms' separated vocals and downstream runs were written.
WORK = Path(os.environ.get("FINESUB_RATE_WORK", "out/separator-rate"))

MATERIALS = [
    "BV1UBjq6fEgb",
    "BV1kYLR6AEXv",
    "BV1ySjz6FEzD",
    "BV1cqLR6hEp3",
    "BV1dwjP6LECU",
]

#: Human-timed subtitles, matched to the materials by duration. The text is a
#: Chinese translation and unusable for CER; the timeline is what makes them a
#: referee independent of every separator arm.
GOLD = {
    "BV1kYLR6AEXv": "看立绘.srt",
    "BV1UBjq6fEgb": "新皮肤(2).srt",
    "BV1ySjz6FEzD": "看莉奈娅pv.srt",
    "BV1cqLR6hEp3": "最后的遗产(1).srt",
    "BV1dwjP6LECU": "布伦妮pv(1).srt",
}


def asset(name: str) -> Path:
    return DATA_ROOT / "assets" / "bilibili" / name


def gold(material: str) -> Path:
    return DATA_ROOT / "data" / "manually-refined-subs" / "四月一日" / GOLD[material]


def reference(material: str) -> Path:
    """The LLM-corrected transcript of a 44.1 kHz production run.

    Biased toward any arm close to that run, which is exactly why the null
    control exists -- read the arms against each other, never the absolute CER.
    """
    return DATA_ROOT / "out" / "reference" / material / f"{material}-corrected.srt"
