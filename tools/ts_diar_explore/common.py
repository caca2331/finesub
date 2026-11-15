"""Shared plumbing for the timestamp + diarization model survey (its write-up is in the maintainer's local notes).

Every model adapter reduces its raw response to one schema so the scorers never see provider shapes:

    {"model": ..., "config": {...}, "audio": ..., "offset": s, "duration": s,
     "segments": [{"start", "end", "speaker", "text"}],       # speaker is None when not asked for
     "words":    [{"start", "end", "speaker", "text"}] | None} # None when the model has no word timing

Times are absolute seconds on the source file (the clip offset is added back). Raw responses are kept
next to the parsed file so a parser bug can be fixed without spending quota again.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]


def _main_checkout(path: Path) -> Path:
    parts = path.parts
    return Path(*parts[: parts.index(".worktrees")]) if ".worktrees" in parts else path


MAIN = _main_checkout(WT)
TMP = WT / "tmp" / "ts-diar"
#: every arm's raw responses (paid ones included) and outputs -- not scratch, so under the main checkout's data/,
#: shared by every worktree like the quota ledger
EPISODES = MAIN / "data" / "ts-diar" / "episodes"
#: Quota state lives in the main checkout: a worktree's ignored tmp/ vanishes with `git worktree remove`, and a
#: run from dev after the merge must still see what was spent (mai2's 18 h stop, the day's 35t / Pro counts).
QUOTA = MAIN / "tmp" / "ts-diar-quota"
LEDGER = QUOTA / "ledger.jsonl"
RUNS = MAIN / "out" / "refined-2609"
IDS = MAIN / "tmp" / "refined-2609" / "ids.json"


ALI = MAIN / "data" / "diarization" / "public" / "alimeeting"


def source_audio(item: str) -> Path:
    """The original mix of a refined-2609 item (the .m4a next to the refined .ass); `ali-<session>` is an
    AliMeeting Eval far-field mix (16 kHz mono, the overlap benchmark)."""
    if item.startswith("ali-"):
        return ALI / "audio" / f"{item[4:]}.wav"
    if item.startswith("gold-"):
        return vocal_track(item)
    if item.startswith("manosaba-"):  # the 魔裁 story clips' recordings, as their production run was fed
        return MAIN / "assets" / "manosaba" / f"{item}.mp4"
    return Path(json.loads(IDS.read_text(encoding="utf-8"))[item]["source"])


def vocal_track(item: str) -> Path:
    """What every arm is fed: production's separated vocal track; for AliMeeting the mix itself (meeting
    speech, no music to separate)."""
    if item.startswith("ali-"):
        return source_audio(item)
    if item.startswith("gold-"):  # a clip of the production baseline (hand-timed / gold-labelled material)
        clip_id = item[5:]
        return MAIN / "out" / "ts-baseline-20260831-hq" / clip_id / f"{clip_id}-vocal.ogg"
    if item.startswith("onetonton-"):  # 姐姐咚咚 runs live in out/<item>/, not under refined-2609
        return MAIN / "out" / item / f"{item}-vocal.ogg"
    if item.startswith("manosaba-"):  # 魔裁 story clips (FlowOCR nameplate truth), production run in out/manosaba/
        return MAIN / "out" / "manosaba" / f"{item}-vocal.ogg"
    return RUNS / item / f"{item}-vocal.ogg"


def clip(src: Path, out: Path, start: float = 0.0, duration: float | None = None) -> Path:
    """16 kHz mono clip. Suffix picks the codec (.wav / .flac)."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-ac", "1", "-ar", "16000", str(out)]
    subprocess.run(cmd, check=True, encoding="utf-8", errors="replace")
    return out


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    return float(out.stdout.strip())


def ledger(**entry) -> None:
    """One line per paid / quota-bearing call, success or not."""
    TMP.mkdir(parents=True, exist_ok=True)
    entry = {"at": _dt.datetime.now().isoformat(timespec="seconds"), **entry}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def mai2_hours_used() -> float:
    return sum(r.get("audio_seconds", 0.0) for r in ledger_rows() if r.get("model") == "mai2") / 3600


def quota_gate(model: str) -> None:
    """Refuse while the owner's lock is in place: the provider's quota day is not ours to guess."""
    lock = QUOTA / f"quota-lock-{model}"
    if lock.exists():
        raise SystemExit(f"{lock.name}: {lock.read_text(encoding='utf-8').strip()}")


#: Where each provider's quota day starts at midnight (owner 2026-10-30: MOSS Pro at Beijing midnight, after the
#: owner logs in to its console; 35t at Pacific midnight).
RESET_TZ = {"g35t": "America/Los_Angeles", "mosspro": "Asia/Shanghai"}


def quota_day_start(model: str, now: _dt.datetime | None = None) -> str:
    """The current quota day's first moment as a local naive ISO time, the ledger's `at` form."""
    from zoneinfo import ZoneInfo

    now = (now or _dt.datetime.now().astimezone()).astimezone(ZoneInfo(RESET_TZ[model]))
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.astimezone().replace(tzinfo=None).isoformat(timespec="seconds")


def quota_day_rows(model: str, label: str | None = None) -> list[dict]:
    """Ledger rows of the current quota day."""
    start = quota_day_start(model)
    return [r for r in ledger_rows()
            if r.get("model") == model and r["at"] >= start and (label is None or r.get("key") == label)]


def g35t_calls_today(label: str | None = None) -> int:
    return len(quota_day_rows("g35t", label))


def save(parsed: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parsed, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def words_to_segments(words: list[dict], gap: float = 0.6) -> list[dict]:
    """Merge consecutive words sharing a speaker, split at pauses > gap. Only for eyeballing:
    the segmentation arms of the survey build their own lines."""
    segs: list[dict] = []
    for w in sorted(words, key=lambda w: w["start"]):
        if segs and segs[-1]["speaker"] == w["speaker"] and w["start"] - segs[-1]["end"] <= gap:
            segs[-1]["end"] = max(segs[-1]["end"], w["end"])
            segs[-1]["text"] += w["text"]
        else:
            segs.append(dict(w))
    return segs


def frozen_inputs(manifest, data):
    """Every input a readout reads from the main checkout's tmp/ and out/ must be its frozen copy under
    `data`/inputs/ (listed with its hash in `manifest`, at its path there): a missing one is put back from the copy;
    one that differs stops the readout -- a rerun of production, or a correction plan regenerated by
    `window_judge_accept.fast`, would change the readouts silently."""
    differ = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, path = line.partition("  ")
        if not path.startswith("inputs/"):
            continue
        live = MAIN / path.removeprefix("inputs/")
        if not live.exists():
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data / path, live)
        elif hashlib.sha256(live.read_bytes()).hexdigest() != digest:
            differ.append(str(live))
    if differ:
        raise SystemExit(f"these inputs differ from their frozen copies under {data / 'inputs'} "
                         "(move them aside to get the copies back):\n  " + "\n  ".join(differ))
