"""Trials for speaker-aware merging on D6 windows (docs/reports/2026-10-17-three-pass.md).

    python tools/diarization_explore/merge_trial.py judged sol             # D6 stables with Sol's verdicts applied
    python tools/diarization_explore/merge_trial.py run <arm> <key>...      # correct the windows of one arm

Arms: `acoustic` reads the D6 stables as they are (acoustic tracks); `judged-sol` reads the copies `judged` wrote,
whose moved rows lead the judge's track with no `?`, as `judgments.judged_speakers` reads them. Runs go to
tmp/spk-merge/<arm>[-<label>]/<key>/ with TRIAL_MODEL (default gemini 3.7 flash, free tier), text only, the acceptance
runs' research context, the code of TRIAL_CHECKOUT (default: the checkout this script sits in). Compare with `merge_audit.py compare <folder>...`.
TRIAL_JUDGE_FROM=<run folder name> reuses that run's judge (its input/ copy of the stable with the judge's sidecar) instead
of judging again; TRIAL_RULES=relaxed corrects through relaxed_rules.py (the owner's merge principles, no hard threshold).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CHECKOUT = HERE.parents[2]
MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
D6 = MAIN / "tmp/row-split/d6"
OUT = MAIN / "tmp/spk-merge"
PRODUCTS = MAIN / "data/diarization/row-truth/products"
MODEL = os.environ.get("TRIAL_MODEL", "gemini-free-3_7-flash")  # a route target for --llm-model
#: the checkout whose code runs the correction (TRIAL_CHECKOUT, e.g. the main checkout for a v93 baseline)
RUN_CHECKOUT = Path(os.environ.get("TRIAL_CHECKOUT", str(CHECKOUT)))
#: TRIAL_JUDGE=text|audio: production's speaker judge runs first (on a copy of the stable) and the correction reads its turns
JUDGE = os.environ.get("TRIAL_JUDGE", "")
JUDGE_FROM = os.environ.get("TRIAL_JUDGE_FROM", "")
RULES = os.environ.get("TRIAL_RULES", "")


def judged(judge):
    """Copy every D6 stable with the full file's verdicts (`products/<judge>/<ident>`) applied to its rows."""
    sys.path.insert(0, str(CHECKOUT / "src"))
    from finesub.speech.diarization.judgments import read_judgments
    from finesub.subtitles.stable_rows import read_rows

    for folder in sorted((D6 / "orig").iterdir()):
        key = folder.name
        ident = key.rsplit("-c", 1)[0]
        full = PRODUCTS / judge / ident / f"{ident}-stable.json"
        verdicts = read_judgments(full)
        if verdicts is None:
            print(f"{key}: no verdicts in {full}")
            continue
        times = {str(row.get("id") or n): (round(row["start"], 3), round(row["end"], 3))
                 for n, row in enumerate(read_rows(full), start=1)}
        moved = {times[key_]: track for key_, track in verdicts.moved.items()}
        payload = json.loads((folder / f"{key}-stable.json").read_text(encoding="utf-8"))
        count = 0
        for segment in payload["segments"]:
            track = moved.get((round(segment["start"], 3), round(segment["end"], 3)))
            if track is None:
                continue
            old = segment.get("speaker") or {}
            segment["speaker"] = {"labels": old.get("labels", []), "block": old.get("block", 0), "track": track,
                                  "tracks": [track], "coverage": {str(track): 1.0}}
            count += 1
        target = OUT / f"stables-judged-{judge}" / key / f"{key}-stable.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"{key}: {count} rows moved")


def stable_for(arm, key):
    if arm == "acoustic":
        return D6 / "orig" / key / f"{key}-stable.json"
    judge = arm.removeprefix("judged-")
    return OUT / f"stables-judged-{judge}" / key / f"{key}-stable.json"


def run(arm, keys, label=""):
    folder = OUT / (f"{arm}-{label}" if label else arm)
    env = {**os.environ, "PYTHONPATH": str(RUN_CHECKOUT / "src")}
    for key in keys:
        ident = key.rsplit("-c", 1)[0]
        target = folder / key
        if list(target.glob("*-annotated.csv")):
            print(f"{key}: done before")
            continue
        target.mkdir(parents=True, exist_ok=True)
        stable = stable_for(arm, key)
        if JUDGE_FROM:  # the sidecar covers the stable, so production reads it and does not judge again
            shutil.copytree(OUT / JUDGE_FROM / key / "input", target / "input")
            stable = target / "input" / stable.name
        elif JUDGE:  # the judge's sidecar lands next to the stable: give it a copy
            (target / "input").mkdir(exist_ok=True)
            stable = Path(shutil.copyfile(stable, target / "input" / stable.name))
        context = MAIN / "tmp/diarization-retrack/accept" / ident / "llm" / f"{ident}-research-context.json"
        entry = [str(HERE.with_name("relaxed_rules.py"))] if RULES == "relaxed" else ["-m", "finesub.llm.correction_translation"]
        cmd = [sys.executable, *entry, str(stable),
               "--llm-model", MODEL, "--context-file", str(context), "--retrieval", "none", "--media", "text",
               "--fast", "off", "-o", str(target / f"{key}.srt"), "--task-artifact-dir", str(target / "llm"),
               *(["--speaker-judge", JUDGE or "text"] if JUDGE or JUDGE_FROM else []), "--execute"]
        with open(target / "run.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(RUN_CHECKOUT), env=env)
        print(f"{key}: exit {code}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "judged":
        judged(sys.argv[2])
    elif sys.argv[1] == "run":
        label = os.environ.get("TRIAL_LABEL", "")
        run(sys.argv[2], sys.argv[3:], label)
