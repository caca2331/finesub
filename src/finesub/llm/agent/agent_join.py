"""`finesub agent-join`: the prompt that lets a person's own agent serve a run.

docs/llm_local_agent.md §12.1.4. A run whose cell is bound to the
conversational target queues its text calls on an assignment and waits for
somebody's agent to claim them; this command prints the bootstrap the person
pastes into that agent. Nothing is injected anywhere -- handing the prompt
over is the person's act, and the agent's permissions stay the person's.

The root may be left out. A run only announces its own once, from a worker
thread, through the reporter -- which is silent wherever nothing bound one --
and the directory carries a fresh random id every time, so a person who
missed that line or was not yet at the keyboard had no way back to it.
Finding the waiting assignment is something this command can do itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .agent_paths import conversational_assignment_parent
from .agent_task_runtime import AgentTaskRuntime
from .agent_transports import conversational_bootstrap


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finesub agent-join")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        help=(
            "assignment root the run announced (its control/index.json names "
            "the assignment); omit it to join the run that is waiting"
        ),
    )
    parser.add_argument(
        "--worker",
        default="",
        help="worker id to join as (default: conv-<n>, the next free one)",
    )
    return parser


def _is_open(root: Path) -> bool:
    """An assignment still worth joining: readable, and not sealed."""

    try:
        runtime = AgentTaskRuntime(root)
        index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
        state = json.loads(runtime.read_artifact(str(index["state_ref"])))
    except Exception:  # noqa: BLE001 -- a half-written or foreign tree is not ours
        return False
    return not bool(state.get("sealed"))


def waiting_assignments(parent: Path | None = None) -> list[Path]:
    """Every open conversational assignment, most recently started first."""

    directory = parent if parent is not None else conversational_assignment_parent()
    try:
        children = [path for path in directory.iterdir() if path.is_dir()]
    except OSError:
        return []
    found = []
    for path in children:
        if not _is_open(path):
            continue
        try:
            found.append((path.stat().st_mtime, path))
        except OSError:
            # It was there a moment ago; a tree that is being cleaned out from
            # under the scan is simply not one of the runs worth joining.
            continue
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _mtime, path in found]


def _resolve_root(given: Path | None) -> Path:
    if given is not None:
        return given.expanduser()
    waiting = waiting_assignments()
    if not waiting:
        # The scan only sees this installation's own coordination domain. A
        # run started from a different checkout or install keeps its trees
        # somewhere else entirely, so "nothing waiting" and "you are looking
        # in the wrong place" arrive as the same empty directory.
        raise FileNotFoundError(
            "no run is waiting for an agent under "
            f"{conversational_assignment_parent()}. Either the run has not "
            "queued its first call yet (it does that only after the speech "
            "stages finish), or it was started from a different FineSub "
            "install or checkout than this one -- in that case pass the "
            "assignment root it printed."
        )
    if len(waiting) > 1:
        listed = "\n  ".join(str(path) for path in waiting)
        raise ValueError(
            "more than one run is waiting; name the one you mean:\n  " + listed
        )
    return waiting[0]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = _resolve_root(args.root)
        runtime = AgentTaskRuntime(root)
        index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
        assignment_id = str(index["assignment_id"])
        # Reserved, not just read: two people joining the same run before
        # either has claimed anything would otherwise both be told they are
        # `conv-1`, and the second one would resume the first one's task.
        worker_id = runtime.register_worker(
            assignment_id=assignment_id,
            worker_id=args.worker,
            kind="conversational",
        )["worker_id"]
        text = conversational_bootstrap(
            runtime, assignment_id=assignment_id, worker_id=worker_id
        )
    except Exception as exc:  # noqa: BLE001 -- a CLI reports, it does not trace
        print(f"agent-join: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.root is None:
        print(f"Joining {assignment_id} at {root}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
