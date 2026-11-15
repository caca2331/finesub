"""Run the agent session-mode arms (docs/llm_local_agent_experiments.md §3.5).

One "run" = one full ``python -m finesub.llm.correction_translation --execute``
over a frozen stable.json, every text cell bound to one local-agent target
under one ``agent_session`` mode, in its own artifact directory. The binding
and the mode come from a config file written per arm (``FINESUB_CONFIG_FILE``),
because ``agent_session`` has no CLI flag -- it is a preset table. Fixed for
every arm: ``--media text`` (the tool protocol is text-only; a media call runs
its own one-off session under every mode, which would erase the difference
being measured), ``--retrieval none --fast off``; ``--continuity`` is ``serial``
unless ``--continuity parallel`` is given, and ``--knowledge`` is ``none`` unless
``--knowledge read`` is given. ``--retrieval local`` and ``--media audio --audio
<file>`` override the first two for a run that measures the capsule path itself
(a request turn, an agy media window).

Arms are interleaved (bench-discipline: never all of A then all of B) and a
finished run is skipped, so the script can be re-invoked to continue. Every
run writes ``run.json`` next to its outputs; ``score_flow_ab.py`` reads those.

The preparation task can be run once and shared (``--prepare <arm>``, then
``--context-file``): whether one run's preparation happened to recognise a
misheard name decided most of the quality spread in the 2026-09-30 pilot, and
it is the same task under every session mode, so sharing it takes that noise
out of the comparison.

    python -m tools.bench.run_session_ab --stable <x-stable.json> --out out/session-ab/x --prepare agy38f-per-window
    python -m tools.bench.run_session_ab --stable <x-stable.json> --out out/session-ab/x --n 5 \\
        --context-file out/session-ab/x/prepare/<...>research-context.json \\
        --arms agy38f-per-window-capsule agy38f-resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]

TARGETS: Dict[str, str] = {
    "agy38f": "local-agy-gemini-3_8-flash",
    "wbdsv41f": "local-codebuddy-deepseek-v4_1-flash",
    "wbhy3": "local-codebuddy-hy3",
    "wbglm53f": "local-codebuddy-glm-5_3-flash",
    "luna": "local-codex-gpt-6-luna",
    "cchaiku45": "local-claude-haiku",
}
MODES = ("per-window", "resume", "pseudo-conversational")
#: The text cells a correction run can call (retrieval=none, knowledge=none).
CELLS = ("research/quality", "correction-text/quality", "knowledge/quality")
PRESET = "session-ab"


#: ``per-window`` arm suffixes -> the dev-only environment variable they set.
SUFFIXES = {
    # The capsule transport (only ``per-window`` honours it: the other modes
    # have a transport by definition) -- the tool-session vs capsule arm.
    "-capsule": ("FINESUB_AGENT_TRANSPORT", "capsule"),
    # A single-task tool session that calls `next_task` first, as before the
    # opening carried the task (docs/llm_agent_tool_protocol.md §1 "开场带上任务").
    "-plain": ("FINESUB_TOOL_SESSION_OPENING", "0"),
}


def arm_parts(arm: str) -> tuple[str, str, Dict[str, str]]:
    """``<target>-<mode>[-capsule|-plain]`` -> (route target, session mode, env)."""

    target, _, rest = arm.partition("-")
    env: Dict[str, str] = {}
    for suffix, (name, value) in SUFFIXES.items():
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)]
            env[name] = value
    if target not in TARGETS or rest not in MODES or (env and rest != "per-window"):
        raise ValueError(
            f"unknown arm {arm!r}: <{'|'.join(TARGETS)}>-<{'|'.join(MODES)}>, "
            f"or per-window{'|'.join(SUFFIXES)}"
        )
    return TARGETS[target], rest, env


def config_text(
    target: str,
    mode: str,
    max_window_subtitle_tokens: int | None,
    *,
    keep_episodes: bool = False,
    media: str = "text",
) -> str:
    # A media window runs on the multimodal correction cell.
    cells = (*CELLS, "correction-mm/quality") if media != "text" else CELLS
    lines = [
        "[llm]",
        f'preset = "{PRESET}"',
        'execution_policy = "agent-only"',
        *(["local_agent_keep_episodes = true"] if keep_episodes else []),
        "",
        f"[llm.presets.{PRESET}]",
        f'name = "Session A/B: {target} {mode}"',
        "",
        f"[llm.presets.{PRESET}.bindings]",
        *(f'"{cell}" = "{target}"' for cell in cells),
        "",
        f"[llm.presets.{PRESET}.agent_session]",
        *(f'"{cell}" = "{mode}"' for cell in cells),
    ]
    if max_window_subtitle_tokens is not None:
        lines += ["", "[chunking]", f"max_window_subtitle_tokens = {int(max_window_subtitle_tokens)}"]
    return "\n".join(lines) + "\n"


def command(
    stable: Path,
    out_srt: Path,
    artifact_dir: Path,
    *,
    execute: bool,
    context_file: Path | None = None,
    research_only: bool = False,
    knowledge: str = "none",
    continuity: str = "serial",
    retrieval: str = "none",
    media: str = "text",
    audio: Path | None = None,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "finesub.llm.correction_translation", str(stable),
        "--media", media, "--retrieval", retrieval, "--knowledge", knowledge, "--fast", "off",
        "--continuity", continuity, "--difficulty", "quality",
        "--task-artifact-dir", str(artifact_dir), "-o", str(out_srt),
    ]
    if audio is not None:
        cmd += ["--audio", str(audio)]
    if context_file is not None:
        cmd += ["--context-file", str(context_file)]
    if research_only:
        cmd.append("--research-only")
    if execute:
        cmd.append("--execute")
    else:
        cmd += ["--prompt-dir", str(artifact_dir / "prompts")]
    return cmd


def run_one(
    *,
    arm: str,
    index: int,
    stable: Path,
    out_root: Path,
    max_window_subtitle_tokens: int | None = None,
    execute: bool = True,
    force: bool = False,
    keep_episodes: bool = False,
    context_file: Path | None = None,
    research_only: bool = False,
    knowledge: str = "none",
    continuity: str = "serial",
    retrieval: str = "none",
    media: str = "text",
    audio: Path | None = None,
) -> Dict[str, object]:
    target, mode, arm_env = arm_parts(arm)
    # A finished run is skipped by its directory, so continuity is part of it.
    arm_dir = arm if continuity == "serial" else f"{arm}-{continuity}"
    run_dir = out_root / "prepare" / arm if research_only else out_root / arm_dir / f"run{index:02d}"
    meta_path = run_dir / "run.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("exit_code") == 0:
            return {"skipped": True, **meta}
    run_dir.mkdir(parents=True, exist_ok=True)
    config = run_dir / "config.toml"
    config.write_text(
        config_text(target, mode, max_window_subtitle_tokens, keep_episodes=keep_episodes, media=media),
        encoding="utf-8",
    )
    artifact_dir = run_dir / "artifacts"
    out_srt = run_dir / f"{stable.stem.replace('-stable', '')}.srt"
    cmd = command(
        stable, out_srt, artifact_dir,
        execute=execute, context_file=context_file, research_only=research_only,
        knowledge=knowledge, continuity=continuity, retrieval=retrieval, media=media, audio=audio,
    )
    env = dict(os.environ)
    env["FINESUB_CONFIG_FILE"] = str(config)
    for name, _value in SUFFIXES.values():
        env.pop(name, None)
    env.update(arm_env)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    started = time.time()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    meta = {
        "arm": arm,
        "index": index,
        "target": target,
        "agent_session": mode,
        "transport": arm_env.get("FINESUB_AGENT_TRANSPORT", "derived"),
        "opening": arm_env.get("FINESUB_TOOL_SESSION_OPENING") != "0",
        "knowledge": knowledge,
        "retrieval": retrieval,
        "media": media,
        "execute": execute,
        "max_window_subtitle_tokens": max_window_subtitle_tokens,
        "keep_episodes": keep_episodes,
        "context_file": str(context_file) if context_file else "",
        "research_only": research_only,
        "stable": str(stable),
        "command": cmd,
        "started": started,
        "wall_seconds": round(time.time() - started, 1),
        "exit_code": proc.returncode,
        "repo": str(REPO),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stable", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--start", type=int, default=1, help="first run index (to add runs later)")
    parser.add_argument("--arms", nargs="+", help="<target>-<mode>, e.g. agy38f-resume")
    parser.add_argument("--max-window-subtitle-tokens", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true", help="dry run: plan windows, no model calls")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-episodes", action="store_true",
        help="[llm] local_agent_keep_episodes: keep every agent episode (and Codex's rollout) for inspection",
    )
    parser.add_argument(
        "--prepare", metavar="ARM",
        help="run only the preparation task once under this arm, into <out>/prepare/<arm>, and print its context file",
    )
    parser.add_argument("--knowledge", choices=("none", "read"), default="none")
    parser.add_argument("--continuity", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--retrieval", choices=("none", "local"), default="none")
    parser.add_argument("--media", choices=("text", "audio"), default="text")
    parser.add_argument("--audio", help="the source audio, with --media audio")
    parser.add_argument(
        "--context-file",
        help="share one preparation result (research-context.json) across every run; skips the preparation task",
    )
    args = parser.parse_args(argv)
    if args.prepare:
        arms = [args.prepare]
    elif args.arms:
        arms = args.arms
    else:
        parser.error("--arms is required unless --prepare is given")
    for arm in arms:
        arm_parts(arm)
    if (args.media == "audio") != bool(args.audio):
        parser.error("--media audio and --audio go together")
    stable = Path(args.stable).resolve()
    out_root = Path(args.out).resolve()
    source = dict(
        retrieval=args.retrieval, media=args.media,
        audio=Path(args.audio).resolve() if args.audio else None,
    )
    if args.prepare:
        meta = run_one(
            arm=args.prepare, index=1, stable=stable, out_root=out_root,
            max_window_subtitle_tokens=args.max_window_subtitle_tokens,
            force=args.force, keep_episodes=args.keep_episodes, research_only=True,
            knowledge=args.knowledge, **source,
        )
        found = sorted((out_root / "prepare" / args.prepare).rglob("*research-context*.json"))
        print(f"prepare {args.prepare}: exit {meta.get('exit_code')}; context file: "
              f"{found[0] if found else 'not found'}", flush=True)
        return 0 if found else 1
    context_file = Path(args.context_file).resolve() if args.context_file else None
    if context_file is not None and not context_file.is_file():
        parser.error(f"--context-file {context_file} does not exist")
    for index in range(args.start, args.start + args.n):
        for arm in args.arms:
            meta = run_one(
                arm=arm, index=index, stable=stable, out_root=out_root,
                max_window_subtitle_tokens=args.max_window_subtitle_tokens,
                execute=not args.plan_only, force=args.force, keep_episodes=args.keep_episodes,
                context_file=context_file, knowledge=args.knowledge,
                continuity=args.continuity, **source,
            )
            state = "skipped" if meta.get("skipped") else f"exit {meta['exit_code']}"
            print(f"{arm} run{index:02d}: {state} ({meta['wall_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
