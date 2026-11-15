"""Run the knowledge-slimming A/B (maintainers only).

    python tools/kb_phonetic_eval/run_ab.py prep --root <dir> --clips id,id --model <target> [--execute]
    python tools/kb_phonetic_eval/run_ab.py arm  --root <dir> --clips id,id --model <target> --name A1 \
        [--slim] [--execute] [--force]

What costs money: only ``--execute``. Without it both stages are dry-runs that write the plan and the
prompts (no research context, no subtitles). With it:
  prep  one preparation task per clip; a clip whose research context already exists is skipped.
  arm   one correction run per clip, ``--parallel`` at a time.
``--model`` is a routing target id of finesub (for instance ``local-codebuddy-deepseek-v4_1-flash``).

Inputs, all under <root> unless said otherwise:
  kb/knowledge.sqlite   the knowledge store every stage reads (``strip_leak.py`` makes it), never written
  clips.csv             rows ``clip id | refined subtitles | stable json | note`` (paths relative to the
                        main checkout); the stable is copied once into each clip's dir, the note goes to
                        ``--extra-info``. ``score.py --index`` reads the same file.

Outputs: prep/<id>/ (the research context every arm reuses, so the carried entries are the same in all
arms) and <name>/<id>/ (``<id>-translated.srt``, prompts, artifacts with the exchanges, run.log, and
``arm.json`` recording the switch and model). ``--slim`` runs with ``FINESUB_KB_SLIM=on`` (the production
gate, slimming and hints; each window's record is ``kb_slim`` in its ``correction_window_response``);
without it the arm is production as shipped. The reading libraries must be installed (the ``kb-phonetic``
extra), or the switch falls back to whole entries. Re-running an existing arm name refuses unless
``--force``: a run under the same task id could otherwise reuse the old arm's products.

The arms of the 2026-10 rounds ran the earlier JSON switch (``FINESUB_KB_PHONETIC_HINTS``, with a
calibrated noisy-node list, threshold 90 and the ``entries`` modes); their ``hints.json`` is a record only.

Each clip runs as ``python -m finesub.llm.correction_translation`` in the main checkout (where the
source material lives) with this checkout's ``src`` first on PYTHONPATH -- an editable install of
another checkout would otherwise be imported silently.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[2] / "src"
MAIN = HERE.parents[2]
if MAIN.parent.name == ".worktrees":
    MAIN = MAIN.parents[1]  # source material lives in the main checkout


def read_index(path: Path) -> dict[str, dict[str, str]]:
    """clips.csv: ``clip id | refined subtitles | stable json | note``, paths relative to the main checkout."""

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        clip, refined, stable, note = (col.strip() for col in line.split("|", 3))
        out[clip] = {"refined": refined, "stable": stable, "note": note}
    return out


def entry_names(context: dict) -> list[str]:
    """The entries a research context selected; entries may be names or ``{"key": name}``."""

    names = []
    for key in ("injected_entries", "keep_entries"):
        for item in context.get(key) or []:
            name = item if isinstance(item, str) else str(item.get("key", ""))
            if name:
                names.append(name)
    return sorted(set(names))


def stage_stable(clip: str, args, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    stable = target / f"{clip}-stable.json"
    if not stable.exists():
        shutil.copyfile(MAIN / args.index[clip]["stable"], stable)
    return stable


def run(cmd: list[str], env_extra: dict[str, str], log: Path) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.pop("FINESUB_KB_SLIM", None)
    env.update(env_extra)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n" + json.dumps(env_extra, ensure_ascii=False) + "\n\n")
        handle.flush()
        return subprocess.call(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=str(MAIN))


def base_cmd(stable: Path, args, work: Path, clip: str, task: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "finesub.llm.correction_translation", str(stable),
        "--media", "text", "--knowledge", "read", "--knowledge-root", str(args.root / "kb"),
        "--llm-model", args.model, "--extra-info", args.index[clip]["note"],
        "--task-id", task, "--task-artifact-dir", str(work / "artifacts"),
        "--prompt-dir", str(work / "prompts"),
    ]
    return cmd + (["--execute"] if args.execute else [])


def find_context(root: Path, clip: str) -> Path | None:
    found = sorted((root / "prep" / clip).rglob("*research-context.json"))
    return found[0] if found else None


def prep_one(clip: str, args) -> int:
    if find_context(args.root, clip):
        print(f"{clip}: research context exists, skipped")
        return 0
    work = args.root / "prep" / clip
    stable = stage_stable(clip, args, work)
    cmd = base_cmd(stable, args, work, clip, f"kbab-prep-{clip}") + ["--research-only"]
    return run(cmd, {}, work / "run.log")


def arm_one(clip: str, args) -> int:
    work = args.root / args.name / clip
    stable = stage_stable(clip, args, work)
    context = find_context(args.root, clip)
    if context is None:
        raise SystemExit(f"no research context for {clip}; run prep first")
    env = {"FINESUB_KB_SLIM": "on"} if args.slim else {}
    record = {"model": args.model, "slim": args.slim, "execute": args.execute, "research_context": str(context)}
    if args.slim and not entry_names(json.loads(context.read_text(encoding="utf-8"))):
        print(f"warning: {clip}: the research context selected no entries, so there is nothing to slim")
    (work / "arm.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    cmd = base_cmd(stable, args, work, clip, f"kbab-{args.name}-{clip}") + [
        "--context-file", str(context), "--fast", "off",
        "-o", str(work / f"{clip}-translated.srt"),
    ]
    return run(cmd, env, work / "run.log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["prep", "arm"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--slim", action="store_true", help="run with FINESUB_KB_SLIM=on")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run into an existing arm directory")
    args = ap.parse_args()
    args.root = args.root.resolve()  # the runs work in the main checkout; relative paths would move
    clips = [c for c in args.clips.split(",") if c]
    args.index = read_index(args.root / "clips.csv")
    if missing := [c for c in clips if c not in args.index]:
        raise SystemExit(f"not in {args.root / 'clips.csv'}: {missing}")
    if args.stage == "arm":
        if not args.name:
            raise SystemExit("--name is required for an arm")
        if (args.root / args.name).exists() and not args.force:
            raise SystemExit(f"{args.root / args.name} exists; pick a new --name or pass --force")
    job = prep_one if args.stage == "prep" else arm_one
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        codes = dict(zip(clips, pool.map(lambda clip: job(clip, args), clips)))
    print(json.dumps(codes))
    if any(codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
