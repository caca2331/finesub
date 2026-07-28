"""CLI orchestration for ``python -m tools.session_replay``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from llm.client import VALIDATION_BASE_TEMPERATURE
from .registry import get_session, list_sessions

DEFAULT_RUN = Path("out/reference/BV1ojjc6MEAs")
DEFAULT_CHUNK = "0001"
DEFAULT_SUCCESS_TARGET = 3
DEFAULT_MAX_ATTEMPTS = 9


def resolve_sampling_plan(
    model: str | None,
    *,
    n: int | None = None,
    max_attempts: int | None = None,
) -> tuple[int, int]:
    """Resolve the prompt-iteration sampling policy for a pinned model."""

    model_id = (
        (model or "").strip().lower().removeprefix("gemini/").removeprefix("gemini-")
    )
    if model_id in {"3.6-flash", "3.5-flash"}:
        default_n, default_attempts = 2, 5
    elif model_id == "3.5-flash-lite":
        default_n, default_attempts = 3, 10
    else:
        default_n, default_attempts = DEFAULT_SUCCESS_TARGET, DEFAULT_MAX_ATTEMPTS

    resolved_n = default_n if n is None else n
    resolved_attempts = default_attempts if max_attempts is None else max_attempts
    if resolved_n <= 0:
        raise ValueError("-n must be positive")
    if resolved_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if resolved_attempts < resolved_n:
        raise ValueError("--max-attempts cannot be smaller than -n")
    return resolved_n, resolved_attempts


def run_session_replay(
    session: str,
    *,
    run: Path,
    chunk_id: str,
    out: Path | None = None,
    label: str = "baseline",
    note: str = "",
    n: int | None = None,
    max_attempts: int | None = None,
    dry_run: bool = False,
    test_profile: bool = False,
    force_extract: bool = False,
    thinking_level: str | None = None,
    profile: str | None = None,
    temperature: float = VALIDATION_BASE_TEMPERATURE,
    model: str | None = None,
    force_tier: str | None = None,
    variant: str | None = None,
    loop_version: str = "v1",
    fixture_override: Path | None = None,
) -> Any:
    n, max_attempts = resolve_sampling_plan(
        model, n=n, max_attempts=max_attempts
    )
    print(
        f"sampling: target_successes={n} max_attempts={max_attempts} "
        f"model={model or 'endpoint-chain'}",
        flush=True,
    )
    adapter = get_session(session)
    from .fixture import resolve_run_layout

    stem = resolve_run_layout(Path(run))["run_dir"].name
    out_dir = (
        Path(out).expanduser().resolve()
        if out
        else Path("out/prompt-iterate") / f"{stem}-{chunk_id}" / label
    )
    return adapter.run(
        run=Path(run),
        chunk_id=chunk_id,
        out_dir=out_dir,
        n=n,
        max_attempts=max_attempts,
        label=label,
        note=note,
        dry_run=dry_run,
        test_profile=test_profile,
        force_extract=force_extract,
        thinking_level=thinking_level,
        profile=profile,
        temperature=temperature,
        model=model,
        force_tier=force_tier,
        variant=variant,
        loop_version=loop_version,
        fixture_override=fixture_override,
    )


def build_parser() -> argparse.ArgumentParser:
    sessions = list_sessions()
    parser = argparse.ArgumentParser(
        prog="python -m tools.session_replay",
        description=(
            "Replay a harness session with frozen prior-stage injections. "
            "Correction R2 reuses the full rendered search+extract body from "
            "an existing run (never re-calls the search agent when a fixture exists)."
        ),
    )
    parser.add_argument(
        "session",
        nargs="?",
        default="correction",
        choices=sorted(sessions),
        help="Session kind to hijack (default: correction).",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help=f"Run directory (default: {DEFAULT_RUN}).",
    )
    parser.add_argument(
        "--chunk",
        default=DEFAULT_CHUNK,
        help=f"Window chunk id (default: {DEFAULT_CHUNK}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root; default out/prompt-iterate/<stem>-<chunk>/<label>/",
    )
    parser.add_argument("--label", default="baseline", help="Iterate label / subdir name.")
    parser.add_argument(
        "--note",
        default="",
        help="本轮改动重点（写入 summary.md）。",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help=(
            "Number of validation-ok replies. Model-aware default: 2 for "
            "3.6/3.5 Flash; 3 for 3.5 Flash Lite; otherwise 3."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Max API attempts. Model-aware default: 5 for 3.6/3.5 Flash; "
            "10 for 3.5 Flash Lite; otherwise 9."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rebuild prompts from fixture only; do not call the generation API.",
    )
    parser.add_argument(
        "--test-profile",
        action="store_true",
        help="Use LiteLLMRoleClient test_profile (free-lite endpoint).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract fixture from the R2 exchange even if one exists on disk.",
    )
    parser.add_argument(
        "--thinking-level",
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help=(
            "Override Gemini thinkingLevel for this replay "
            "(default: role/profile config, usually medium)."
        ),
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=[
            "text-low",
            "text-med",
            "text-high",
            "mm-low",
            "mm-med",
            "mm-high",
        ],
        help=(
            "Override fixture profile for this replay (rebuilds prompts; "
            "mm-low/text-* skip media upload)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Pin the endpoint chain to one FREE model. Prefer an exact short "
            "id (e.g. '3.6-flash', '3.5-flash', '3.5-flash-lite'); an "
            "ambiguous fuzzy value is rejected. "
            "For capable acceptance runs prefer 3.6-flash; use 3.5-flash only "
            "after 3.6 quota exhaustion or when explicitly requested. "
            "Success counts become per-model; quota exhaustion aborts with a "
            "report instead of falling back. Mutually exclusive with "
            "--test-profile."
        ),
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Force a named correction prompt variant (registry key: basicA, "
            "capableB, capableC, basicB) regardless of the answering endpoint's tier. "
            "Supersedes --force-tier; reply meta still reports the real tier. "
            "Correction round only — other rounds have no variant set and "
            "raise rather than silently ignore it."
        ),
    )
    parser.add_argument(
        "--force-tier",
        default=None,
        choices=["capable", "basic"],
        help=(
            "Legacy shortcut: force the named tier's DEFAULT variant "
            "(capable->capableC, basic->basicA) regardless of the answering "
            "endpoint's capability (e.g. run the capable prompt on flash-lite). "
            "Prefer --variant for a specific set. Correction round only."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=VALIDATION_BASE_TEMPERATURE,
        help=(
            "Base sampling temperature. Each later model call decreases it by 0.01, "
            "including after validation-ok replies."
        ),
    )
    parser.add_argument(
        "--loop-version",
        default="v1",
        choices=["v1", "v2"],
        help=(
            "Search loop prompt version: v1 (binary continue/pack) or "
            "v2 (always pack, optional queries). Search-judge only."
        ),
    )
    parser.add_argument(
        "--fixture-override",
        type=Path,
        default=None,
        help=(
            "Load fixture directly from this path instead of the default "
            "lookup (session-fixtures/ or exchange extraction). Correction "
            "round only."
        ),
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List registered session kinds and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_sessions:
        for name, desc in list_sessions().items():
            print(f"{name}\t{desc}")
        return 0
    result = run_session_replay(
        args.session,
        run=args.run,
        chunk_id=args.chunk,
        out=args.out,
        label=args.label,
        note=args.note,
        n=args.n,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        test_profile=args.test_profile,
        force_extract=args.force_extract,
        thinking_level=args.thinking_level,
        profile=args.profile,
        temperature=args.temperature,
        model=args.model,
        force_tier=args.force_tier,
        variant=args.variant,
        loop_version=args.loop_version,
        fixture_override=args.fixture_override,
    )
    print(f"summary: {result.summary_path}")
    print(f"out_dir: {result.out_dir}")
    if result.dry_run:
        print("dry-run: prompts written; no API calls")
    else:
        print(f"successes: {len(result.successes)}")
        print(f"failures: {len(result.failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
