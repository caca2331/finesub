"""Acceptance runs for three passes in production (docs/reports/2026-10-17-three-pass.md 「验收的指标与门槛」).

    python tools/diarization_explore/accept_3pass.py run <arm> <round> <key>...
    python tools/diarization_explore/accept_3pass.py dry <arm> <key>...       # prompts only, no call

Every arm is production's `finesub.llm.correction_translation` on a D6 window's stable, on agy Gemini 3.8 Flash
(TRIAL_MODEL), with the acceptance runs' research context, no retrieval, regular windows, the window's recording
(the separated vocals: the prototype's runs heard the same) and, for the judge, the vocals beside the stable.
Every arm runs this worktree's code as committed (TRIAL_COMMIT, default HEAD), from a snapshot under
<out>/code-<commit>/, so the worktree can change while runs go. A baseline is production's one pass at the commit run (v95 at the 2026-10-17 acceptance; the judge's turns): its task
directory is seeded with a recorded one-pass choice (`correction-mode.json`), which a run follows as a resumed
task does. `text-p2` reads
the `main` arm's pass 1 of the same round (its sidecar is copied, so the judge does not run again) and corrects on
text. `judge-audio-only` is the audio arm's yardstick for the judge's turns: production's `judge-audio` without
the shape section, run alone -- a copy of the `base-judge` arm's finished task of the same round with its sidecar
removed, so the run takes the finished-task path and only the judge runs (the correction is not redone).
Runs go to tmp/spk-merge/accept/<arm>-r<round>/<key>/; `accept_3pass_readout.py` reads them.
PASS1_ONLY=1 stops a three-pass arm once its judge is done (every sidecar window settled for SETTLE seconds), ending
the run's process tree: for reading the judge alone (docs/plans/argon-audio-judge-plan.md); such a run has no
`-annotated.csv`, so it is redone unless its sidecar is there. TRIAL_VOCAL=silence swaps the recording for silence.
The `trackless` arm needs TRIAL_COMMIT set to a commit before the trackless pass 1 was deleted (merge of
`difficulty-prompts`): at a later commit a task without the judge runs one pass, so the arm refuses to start.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[2]
MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
D6 = MAIN / "tmp/row-split/d6/orig"
ACCEPT = MAIN / "tmp/diarization-retrack/accept"
OUT = MAIN / "tmp/spk-merge/accept"
#: materials made for the acceptance (two consecutive D6 windows as one stable: hina-22-c0102), else D6's
MATERIALS = OUT / "materials"
MODEL = os.environ.get("TRIAL_MODEL", "local-agy-gemini-3_8-flash")

#: arm -> (one pass, --speaker-judge value or None, media, arm whose pass 1 it reads)
ARMS = {
    "base-judge": (True, "text", "audio", None),        # one pass: the judge, then v95 on its turns
    "base-plain": (True, None, "audio", None),          # one pass without a judge: v95 on acoustic turns
    "base-audio-judge": (True, "audio", "audio", None),  # one pass with judge-audio, for the audio arm's marks
    "main": (False, "text", "audio", None),             # three passes, pass 1 tracked
    "main-text": (False, "text", "text", None),         # the same on text (a target without audio: Sol)
    "trackless": (False, None, "audio", None),          # three passes, pass 1 trackless (`--diarize acoustics`)
    "audio": (False, "audio", "audio", None),           # three passes, pass 1 with audio (split / fill / insert)
    "text-p2": (False, "text", "text", "main"),         # the main arm's pass 1, pass 2 and 3 on text
    "judge-audio-only": (True, "audio", "audio", "base-judge"),  # the judge alone on base-judge's finished task
}
PASS1_ONLY = os.environ.get("PASS1_ONLY") == "1"
#: TRIAL_VOCAL=silence gives the run a silent recording of the same length (samples included): does the audio judge
#: still move rows when it hears nothing (docs/plans/argon-audio-judge-plan.md 「P2 静音替身」)
SILENT = os.environ.get("TRIAL_VOCAL") == "silence"
SETTLE = 20
ONE_PASS = {"version": 1, "mode": "single", "proofread": False, "reason": "acceptance baseline: production's one pass"}


def _code() -> Path:
    """The committed source the runs import, extracted once per commit."""
    commit = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "--short", os.environ.get("TRIAL_COMMIT", "HEAD")],
                            capture_output=True, text=True, check=True).stdout.strip()
    target = OUT / f"code-{commit}"
    if not (target / "src").is_dir():
        archive = subprocess.run(["git", "-C", str(WORKTREE), "archive", commit, "src"], capture_output=True, check=True)
        partial = target.with_name(f".{target.name}")
        shutil.rmtree(partial, ignore_errors=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(partial, filter="data")
        partial.replace(target)
    return target


def _place(target: Path, key: str, fork: Path | None) -> Path:
    """The stable the run reads, in the run's own input/ (the judge writes its sidecar beside it), with the
    vocals beside it under the name the judge looks for; a fork starts from the other arm's input/."""
    ident = key.rsplit("-c", 1)[0]
    folder = target / "input"
    if fork is not None:
        shutil.copytree(fork / "input", folder, dirs_exist_ok=True)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        source = MATERIALS / key if (MATERIALS / key).is_dir() else D6 / key
        shutil.copyfile(source / f"{key}-stable.json", folder / f"{key}-stable.json")
    vocal = folder / f"{key}-vocal.ogg"
    if not vocal.exists() and SILENT:
        import soundfile as sf
        info = sf.info(str(ACCEPT / ident / f"{ident}-vocal.ogg"))
        # ⚠ not soundfile: its Vorbis encoder kills the process (exit 127, no traceback) on a long silent buffer
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r={info.samplerate}:cl=mono",
                        "-t", f"{info.frames / info.samplerate:.3f}", "-c:a", "libvorbis", str(vocal)], check=True)
    elif not vocal.exists():
        try:
            os.link(ACCEPT / ident / f"{ident}-vocal.ogg", vocal)
        except OSError:
            shutil.copyfile(ACCEPT / ident / f"{ident}-vocal.ogg", vocal)
    return folder / f"{key}-stable.json"


def _context(ident: str) -> Path:
    """The file's research context; manosaba's sits only in the forced-research folder (no acceptance run used it
    before P3), and every arm of a window reads the same one, so arms stay comparable."""
    path = ACCEPT / ident / "llm" / f"{ident}-research-context.json"
    return path if path.exists() else ACCEPT / ident / "llm-unused-forced-research" / path.name


def run(arm: str, keys: list[str], round_: str, *, dry: bool = False) -> None:
    one_pass, judge, media, fork_arm = ARMS[arm]
    if arm == "trackless" and "TRIAL_COMMIT" not in os.environ:
        sys.exit("trackless: set TRIAL_COMMIT to a commit that still has the trackless pass 1 (see the docstring)")
    folder = OUT / (f"{arm}-dry" if dry else f"{arm}-r{round_}")
    for key in keys:
        ident = key.rsplit("-c", 1)[0]
        target = folder / key
        alone = arm == "judge-audio-only"
        # the final -annotated.csv only: three passes leave -pass2-annotated.csv on the way
        judge_only = alone or (PASS1_ONLY and not one_pass)
        if not dry and (_judged(target / "input" / f"{key}-speaker-judge.json") if judge_only
                        else (target / f"{key}-annotated.csv").exists()):
            print(f"{key}: done before")
            continue
        fork = OUT / f"{fork_arm}-r{round_}" / key if fork_arm else None
        if fork is not None and not (fork / f"{key}-annotated.csv" if alone else fork / "input" / f"{key}-speaker-judge.json").exists():
            print(f"{key}: {fork_arm} round {round_} has not finished yet")
            continue
        if alone:
            shutil.copytree(fork, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*-speaker-judge.json", "speaker-judge-input.json", "run.log"))
        stable = _place(target, key, None if alone else fork)
        if one_pass:
            (target / "llm").mkdir(parents=True, exist_ok=True)
            (target / "llm" / "correction-mode.json").write_text(json.dumps(ONE_PASS), encoding="utf-8")
        cmd = [sys.executable, "-m", "finesub.llm.correction_translation", str(stable),
               "--llm-model", MODEL, "--context-file", str(_context(ident)),
               "--retrieval", "none", "--media", media, "--audio", str(stable.with_name(f"{key}-vocal.ogg")),
               "--fast", "off", "-o", str(target / f"{key}.srt"), "--task-artifact-dir", str(target / "llm"),
               *(["--speaker-judge", judge] if judge else []),
               *(["--prompt-dir", str(target / "prompts")] if dry else ["--execute"])]
        code_dir = _code()
        # imports come from the snapshot; data (knowledge, .env, state) resolves through the worktree as before
        env = {**os.environ, "PYTHONPATH": str(code_dir / "src"), "FINESUB_ROOT": str(WORKTREE)}
        with open(target / "run.log", "w", encoding="utf-8") as log:
            log.write(f"code: {code_dir.name}\n")
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(WORKTREE), env=env)
            code = _wait(proc, target / "input" / f"{key}-speaker-judge.json" if judge_only and not alone and not dry else None)
        print(f"{arm} r{round_} {key}: exit {code}", flush=True)


def _judged(sidecar: Path) -> bool:
    """Every window (and part) of the judge's sidecar has an answer or gave up: none still pending."""
    try:
        windows = json.loads(sidecar.read_text(encoding="utf-8"))["windows"]
    except (OSError, ValueError, KeyError):
        return False
    units = [unit for window in windows for unit in (window.get("parts") or [window])]
    return bool(units) and all(unit.get("status") != "pending" for unit in units)


def _wait(proc: subprocess.Popen, sidecar: Path | None):
    """The run's exit code; with a sidecar to watch, end the process tree once the judge has settled."""
    settled = None
    while proc.poll() is None:
        if sidecar is not None and _judged(sidecar):
            settled = settled or time.monotonic()
            if time.monotonic() - settled >= SETTLE:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                proc.wait()
                return "stopped after the judge"
        else:
            settled = None
        time.sleep(2)
    return proc.returncode


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[4:], sys.argv[3])
    elif sys.argv[1] == "dry":
        run(sys.argv[2], sys.argv[3:], "", dry=True)
