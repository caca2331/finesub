"""User-invoked cleanup for failed local-agent evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from finesub_bootstrap.agy_records import remove_project_records_under
from finesub_bootstrap.fsops import remove_tree
from finesub_bootstrap.locks import LockUnavailable, holding_activity_barrier
from .agent_paths import (
    machine_temp_root,
    managed_agent_capsule_parent,
    resolve_agent_episode_location,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # The subcommand a user actually has. The `finesub-agent-clean` script
        # this used to name was dropped with the rest of the root package's
        # console scripts; a source checkout runs
        # `python -m finesub.llm.agent.agent_cleanup`, which argparse would
        # otherwise render as a bare `agent_cleanup.py`.
        prog="finesub agent-clean",
        description="Remove retained evidence from failed local-agent calls.",
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="remove every machine-temp domain and the managed evidence root",
    )
    parser.add_argument(
        "--locate",
        metavar="LOCATOR_OR_FILE",
        help="print the current path of retained evidence from its JSON locator",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="confirm the dangerous --all-domains operation non-interactively",
    )
    return parser


def _existing_targets(all_domains: bool) -> list[Path]:
    current = resolve_agent_episode_location()
    candidates = [current.parent]
    if all_domains:
        temp_root = machine_temp_root()
        candidates = [temp_root, temp_root.with_name(temp_root.name + ".lock")]
        if temp_root not in current.parent.parents and current.parent != temp_root:
            candidates.append(current.parent)
        managed = managed_agent_capsule_parent()
        if managed is not None:
            candidates.append(managed)
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in unique and os.path.lexists(resolved):
            unique.append(resolved)
    return unique


def _remove_target(target: Path) -> None:
    if target.is_dir() and not target.is_symlink():
        remove_tree(target)
    else:
        target.unlink(missing_ok=True)


def _read_locator(value: str) -> dict[str, object]:
    candidate = Path(value).expanduser()
    body = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Agent evidence locator must be a JSON object")
    return parsed


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.locate:
        if args.all_domains or args.force:
            print("--locate cannot be combined with cleanup options", file=sys.stderr)
            return 2
        try:
            from .agent_paths import resolve_evidence_locator

            path = resolve_evidence_locator(_read_locator(args.locate))
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(path)
        return 0 if os.path.lexists(path) else 1
    if args.force and not args.all_domains:
        print("--force is only valid with --all-domains", file=sys.stderr)
        return 2

    location = resolve_agent_episode_location()
    targets = _existing_targets(args.all_domains)
    if not targets:
        print("No retained local-agent evidence in this domain.")
        return 0

    if args.all_domains:
        print("The following local-agent evidence roots will be removed:")
        for target in targets:
            print(f"  {target}")
        print(
            "This cannot prove that agents in every checkout/domain are idle; "
            "stop them before continuing."
        )
        if not args.force:
            if not sys.stdin.isatty():
                print("Use --force to confirm in a non-interactive shell.", file=sys.stderr)
                return 2
            if input("Type DELETE to continue: ").strip() != "DELETE":
                print("Cancelled.")
                return 1

    try:
        with holding_activity_barrier(location.activity_root, timeout=0):
            failures: list[str] = []
            for target in targets:
                try:
                    _remove_target(target)
                    print(f"removed {target}")
                except OSError as exc:
                    failures.append(f"{target}: {exc}")
    except (OSError, LockUnavailable):
        print(
            "FineSub local agents are active in this domain; wait for them to finish.",
            file=sys.stderr,
        )
        return 1

    if args.all_domains:
        # agy registered a project per domain (and per tool slot) and has no
        # unregister command; with the domains gone its records would point
        # at nothing. Directory-owned, so re-registration leftovers go too.
        for record in remove_project_records_under(targets):
            print(f"removed agy project record {record}")

    if failures:
        print("Some evidence roots could not be removed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
