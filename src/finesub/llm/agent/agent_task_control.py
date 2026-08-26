"""JSON CLI for conversational Agent access to :mod:`agent_task_runtime`.

The harness creates assignments; an Agent only resumes, claims, checkpoints,
waits, releases or submits within that existing assignment root.  There is no
keepalive command -- every one of these renews the lease on its way past the
fence.  Every successful command writes exactly one JSON object to stdout.  In
particular ``await-next-task`` emits no progress chatter while it blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .agent_task_runtime import (
    CONVERSATIONAL_WATCH_SECONDS,
    AgentTaskRuntime,
    StaleControlGenerationError,
)
from .agent_validators import VALIDATOR_BUILDERS, runtime_validators


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finesub agent-task")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="assignment root (default: current directory)",
    )
    parser.add_argument("--assignment", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--worker", default="")
    rehydrate = subparsers.add_parser("rehydrate")
    rehydrate.add_argument("--worker", default="")

    claim = subparsers.add_parser("next-task")
    _worker_request_arguments(claim)
    claim.add_argument("--control-generation", required=True, type=int)
    # This CLI is the conversational channel (docs/llm_local_agent.md
    # §12.1.4): a person's agent drives it, so the claim is conversational
    # unless a headless harness test says otherwise.
    claim.add_argument("--kind", choices=("conversational", "headless"), default="conversational")

    wait = subparsers.add_parser("await-next-task")
    wait.add_argument("--worker", required=True)
    wait.add_argument("--wait-token", required=True)

    checkpoint = subparsers.add_parser("checkpoint-progress")
    _leased_arguments(checkpoint)
    _json_input_arguments(checkpoint)

    # No `heartbeat` command on purpose: an Agent has no process of its own
    # between two invocations of this CLI, and a model in the middle of a long
    # turn cannot run a timer, so a keepalive it was told to send would be
    # missed exactly when it mattered. Every command below renews the lease as
    # a side effect, which is a liveness signal the Agent cannot forget.
    release = subparsers.add_parser("release-task")
    _leased_arguments(release)
    release.add_argument("--reason", default="")

    submit = subparsers.add_parser("submit")
    _leased_arguments(submit)
    submit.add_argument("--input-hash", required=True)
    _json_input_arguments(submit)

    # Check an answer against the same validator before spending a submit.
    # No `--request-id` and no `--input-hash`: nothing is handed over, so
    # there is no answer to match and nothing to replay -- a lint may be
    # repeated as often as the agent likes and each run is its own.
    lint = subparsers.add_parser("lint")
    lint.add_argument("--worker", required=True)
    lint.add_argument("--task", required=True)
    lint.add_argument("--lease-generation", required=True, type=int)
    _json_input_arguments(lint)

    search = subparsers.add_parser("web-search")
    _leased_arguments(search)
    search.add_argument("--query", required=True)
    search.add_argument("--guided-query", default="")

    fetch = subparsers.add_parser("web-fetch")
    _leased_arguments(fetch)
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--guided-query", default="")
    return parser


def _worker_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--worker", required=True)
    parser.add_argument("--request-id", required=True)


def _leased_arguments(parser: argparse.ArgumentParser) -> None:
    _worker_request_arguments(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--lease-generation", required=True, type=int)


def _json_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json-file",
        type=Path,
        help="read JSON payload from this file; otherwise read one value from stdin",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help=(
            "read this file as the answer text itself, no JSON quoting -- the "
            "usual shape for an answer written straight to a file"
        ),
    )


def _json_input(args: argparse.Namespace) -> Any:
    # An answer is normally one long string, and JSON-quoting tens of KB of it
    # by hand is a step that can only go wrong. `--text-file` takes the file as
    # that string; `--json-file` and stdin keep the general shape.
    text_file = getattr(args, "text_file", None)
    if text_file is not None:
        if args.json_file is not None:
            raise ValueError("Pass either --json-file or --text-file, not both")
        return text_file.read_text(encoding="utf-8")
    if args.json_file is not None:
        text = args.json_file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    if not text.strip():
        raise ValueError("A JSON payload is required")
    return json.loads(text)


def _dispatch(runtime: AgentTaskRuntime, args: argparse.Namespace) -> dict[str, Any]:
    common = {"assignment_id": args.assignment}
    if args.command == "status":
        return runtime.status(**common, worker_id=args.worker)
    if args.command == "rehydrate":
        return runtime.rehydrate(**common, worker_id=args.worker)
    if args.command == "next-task":
        return _claim(runtime, args)
    if args.command == "await-next-task":
        return runtime.await_next_task(
            **common,
            worker_id=args.worker,
            wait_token=args.wait_token,
            # The Agent runs this as a foreground command inside its own turn,
            # so the bound here is the host's turn/tool timeout, not anything
            # the runtime or the provider would prefer.
            max_wait_seconds=CONVERSATIONAL_WATCH_SECONDS,
        )
    if args.command == "lint":
        return runtime.lint(
            assignment_id=args.assignment,
            task_id=args.task,
            worker_id=args.worker,
            lease_generation=args.lease_generation,
            candidate=_json_input(args),
        )
    leased = {
        **common,
        "task_id": args.task,
        "worker_id": args.worker,
        "lease_generation": args.lease_generation,
        "request_id": args.request_id,
    }
    if args.command == "checkpoint-progress":
        return runtime.checkpoint_progress(**leased, progress=_json_input(args))
    if args.command == "release-task":
        return runtime.release_task(**leased, reason=args.reason)
    if args.command == "submit":
        return runtime.submit(
            **leased,
            input_hash=args.input_hash,
            candidate=_json_input(args),
        )
    if args.command in {"web-search", "web-fetch"}:
        from .agent_retrieval import AgentRetrievalAccess

        retrieval = AgentRetrievalAccess(runtime)
        if args.command == "web-search":
            return retrieval.search(
                **leased,
                query=args.query,
                guided_query=args.guided_query,
            )
        return retrieval.fetch(
            **leased,
            url=args.url,
            guided_query=args.guided_query,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


# A claim carries the control generation it was planned against, and the
# harness bumps that whenever it adds the next window's task. Losing that race
# is normal -- it means work arrived -- so the claim is re-planned against the
# state that won rather than reported as an error the caller would have to
# know how to read. Bounded, because a claim that keeps losing is news itself.
CLAIM_ATTEMPTS = 4


def _claim(runtime: AgentTaskRuntime, args: argparse.Namespace) -> dict[str, Any]:
    """`next-task`, re-planned against a control generation that moved."""

    control_generation = int(args.control_generation)
    for attempt in range(CLAIM_ATTEMPTS):
        try:
            return runtime.next_task(
                assignment_id=args.assignment,
                worker_id=args.worker,
                request_id=(
                    args.request_id if attempt == 0 else f"{args.request_id}-{attempt}"
                ),
                expected_control_generation=control_generation,
                worker_kind=args.kind,
            )
        except StaleControlGenerationError:
            if attempt == CLAIM_ATTEMPTS - 1:
                raise
            control_generation = int(
                runtime.status(assignment_id=args.assignment, worker_id=args.worker)[
                    "control_generation"
                ]
            )
    raise AssertionError("unreachable")


def runtime_for(root: Path) -> AgentTaskRuntime:
    """The assignment under ``root`` with every registered validator, so a
    conversational worker's `submit` is judged by the same function the
    harness named in the manifest."""

    validators: dict[str, Any] = {}
    for validator_id in sorted(VALIDATOR_BUILDERS):
        validators.update(runtime_validators(validator_id))
    return AgentTaskRuntime(root, validators=validators)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(runtime_for(args.root), args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
