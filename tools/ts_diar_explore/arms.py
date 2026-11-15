"""Every arm the survey runs or scores, declared once: its name, where its output lives, what it carries, and
(for model arms) how `episode.py` runs it. Adding a model arm = one `Arm(...)` line here plus its adapter.

Files live under `common.EPISODES/<item>/` (the main checkout's `data/ts-diar/episodes/`):

  model arms       `<name>-<input>[-s<A>-<B>].json` (input `vocal` or `mix`; `-sA-B` for a `--span` run), windowed
                   arms also keep each window's parsed output in the same-named directory as `w<k>.json`
  reference arms   `<name>-diar.json`, a speaker timeline only (`dz_baseline.py`, `n3_baseline.py`)
  prod             production text and word times (`production_stable`) + the DiariZen timeline

    from .arms import ARMS, load, exists
    load("hina-72", "mosspro")               # survey schema: {"words": [...] | None, "segments": [...], ...}
    load("hina-73", "moss-single", span=(1320, 2220))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .common import EPISODES, MAIN, RUNS



@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                     # "model" | "reference" | "production"
    label: str                    # column heading in reports
    text: bool = True             # carries text (CER, overlap text)
    words: bool = False           # word-timed; else only segment edges are real
    # model arms only: how episode.py runs it
    window: float | None = None   # window seconds (None = one request over the whole span)
    overlap: float = 0.0
    module: str = ""              # adapter module under tools/ts_diar_explore
    extra: tuple[str, ...] = field(default=())

    @property
    def windowed(self) -> bool:
        return self.window is not None


ARMS: dict[str, Arm] = {a.name: a for a in (
    Arm("prod", "production", "production", words=True),
    Arm("diarizen", "reference", "DiariZen", text=False),
    Arm("n3", "reference", "n3", text=False),
    Arm("n3vbx", "reference", "n3vbx", text=False),
    # mai2 with diarization fails from ~15 min, so it runs in 14 min windows; without, one request.
    Arm("mai2-diar", "model", "mai2 win", words=True, window=840.0, overlap=60.0, module="mai2", extra=("--diarize",)),
    Arm("mai2-nodiar", "model", "mai2 no-diar", words=True, module="mai2"),
    # 35t's 32768-token output cap ends dense speech at ~540 s.
    Arm("g35t", "model", "35t win", words=True, window=540.0, overlap=60.0, module="g35t"),
    # with diarization a 540 s window can come back with words but no speakers (魔裁, 2026-10-05); 300 s is the safer size
    Arm("g35t-w300", "model", "35t 300 s", words=True, window=300.0, overlap=60.0, module="g35t"),
    Arm("moss", "model", "MOSS win", window=300.0, overlap=60.0, module="moss_run"),
    Arm("moss-single", "model", "MOSS single", module="moss_run"),
    # The same open model served by SGLang-Omni in WSL (`moss_serve.py`), in 15 min blocks: short enough that single
    # passes did not merge people, long enough for few seams.
    Arm("moss15", "model", "MOSS 15 min", window=900.0, overlap=60.0, module="moss_serve", extra=("--engine", "sglang")),
    # MOSS Pro answers up to 60 min in one request; its windowed arm uses the open model's geometry so
    # "same base, short windows + our merging" compares against its own single pass.
    Arm("mosspro", "model", "Pro single", module="mosspro"),
    Arm("mosspro-win", "model", "Pro win", window=300.0, overlap=60.0, module="mosspro"),
    # open MOSS through the in-process fast decoder (static KV + CUDA graph): `mossg` plain greedy (control), `mossr`
    # with the run-time rescues, `mossrv` its labels merged by voice. The decoder and the adopted rescues are production's
    # `--asr moss-local` now (`finesub.speech.joint.moss_local`); these arms ran on the experiment version, with a switch
    # per rule (`moss_fast.py` / `moss_kernels.py` / `moss_rescue.py`), kept in the maintainer's archive, not here.
    Arm("mossg", "model", "MOSS greedy", module="moss_rescue"),
    Arm("mossr", "model", "MOSS rescued", module="moss_rescue"),
    Arm("mossrv", "model", "MOSS rescued, labels merged by voice", module="moss_rescue"),  # explore/moss_local_rescue/consolidate.py
)}


def stem(name: str, *, span: tuple[float, float] | None = None, tag: str = "vocal", rep: int = 1) -> str:
    """`rep` > 1 names a repeat call on the same input (the model's own run-to-run noise)."""
    arm = ARMS[name]
    if arm.kind == "reference" or arm.kind == "production":
        return f"{'diarizen' if arm.kind == 'production' else name}-diar"
    return f"{name}-{tag}" + (f"-s{span[0]:.0f}-{span[1]:.0f}" if span else "") + (f"-r{rep}" if rep > 1 else "")


def path(item: str, name: str, **kw) -> Path:
    return EPISODES / item / f"{stem(name, **kw)}.json"


def window_dir(item: str, name: str, **kw) -> Path:
    return EPISODES / item / stem(name, **kw)


def production_stable(item: str) -> Path | None:
    """Production's `-stable.json` for an item: refined-2609 for the hina axes, the ts-baseline run for the
    gold clips; AliMeeting has no production ASR arm."""
    if item.startswith("gold-"):
        clip = item[5:]
        return MAIN / "out" / "ts-baseline-20260831-hq" / clip / f"{clip}-stable.json"
    if item.startswith("ali-"):
        return None
    return RUNS / item / f"{item}-stable.json"


def exists(item: str, name: str, **kw) -> bool:
    if ARMS[name].kind == "production":
        stable = production_stable(item)
        return bool(stable and stable.exists())
    return path(item, name, **kw).exists()


def _timeline(p: Path) -> list[dict]:
    if not p.exists():
        return []
    tracks = json.loads(p.read_text(encoding="utf-8"))["tracks"]
    return [{"start": t["start"], "end": t["end"], "speaker": f"T{t['track']}", "text": ""} for t in tracks]


def load(item: str, name: str, **kw) -> dict:
    """The arm in the survey schema. Reference arms are timelines (no text); production is stable words for
    text and time plus the DiariZen timeline for speakers, and its subtitle lines under `lines`."""
    arm = ARMS[name]
    if arm.kind == "reference":
        return {"model": name, "words": None, "segments": _timeline(path(item, name))}
    if arm.kind == "production":
        stable = json.loads(production_stable(item).read_text(encoding="utf-8"))["segments"]
        return {"model": "prod", "segments": _timeline(path(item, "diarizen")),
                "words": [{"start": w["start"], "end": w["end"], "speaker": None, "text": w["word"]}
                          for s in stable for w in s.get("words", [])],
                "lines": [{"start": s["start"], "end": s["end"], "text": s.get("text", "")} for s in stable]}
    return json.loads(path(item, name, **kw).read_text(encoding="utf-8"))


def windows(item: str, name: str, **kw) -> list[tuple[float, float, dict]]:
    """A windowed arm's per-window outputs, in time order: (start, end, parsed window)."""
    out = []
    for f in sorted(window_dir(item, name, **kw).glob("w*.json")):
        if not f.name.endswith(".raw.json"):
            w = json.loads(f.read_text(encoding="utf-8"))
            out.append((w["offset"], w["offset"] + w["duration"], w))
    return out
